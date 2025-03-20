import asyncio

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBClassifier

from tg_parser.data.feature_engineering import SinglePostDataframe
from tg_parser.utils import get_comments


MODEL_PATH = "models/embeddings_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
TEXT_FEATURE = "message"
params = {
    "scale_pos_weight": 6.031856810020168,
    "n_estimators": 2636,
    "learning_rate": 0.07991244742788668,
    "max_depth": 4,
    "min_child_weight": 6,
    "subsample": 0.9661872052481643,
    "colsample_bytree": 0.5079831911079782,
    "gamma": 0.7386860467543648,
    "reg_alpha": 0.9624343342450836,
    "reg_lambda": 0.27597830103251764,
}
PREDICT_MODEL = XGBClassifier(
    **params,
    objective="binary:logistic",
    eval_metric="auc",
    booster="gbtree",
    enable_categorical=True,
    early_stopping_rounds=100,
    seed=42,
)
PREDICT_MODEL.load_model("models/predict_model/xgb_clf_best_060024.model")


def tokenized_pytorch_tensors(
    df: pd.DataFrame,
    column_list: list,
) -> Dataset:
    transformers_dataset = Dataset.from_pandas(df)

    def tokenize(model_inputs_batch: Dataset) -> Dataset:
        return tokenizer(
            model_inputs_batch[TEXT_FEATURE],
            padding=True,
            max_length=180,
            truncation=True,
        )

    tokenized_dataset = transformers_dataset.map(
        tokenize, batched=True, batch_size=128
    )
    tokenized_dataset.set_format("torch", columns=column_list)

    columns_to_remove = set(tokenized_dataset.column_names) - set(column_list)
    tokenized_dataset = tokenized_dataset.remove_columns(
        list(columns_to_remove)
    )
    return tokenized_dataset


def hidden_state_from_text_inputs(df) -> pd.DataFrame:
    def pool(hidden_state, mask, pooling_method="cls"):
        if pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif pooling_method == "cls":
            return hidden_state[:, 0, :]  # Возвращает CLS-токен

    def extract_hidden_states(batch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {
            k: v.to(device)
            for k, v in batch.items()
            if k in tokenizer.model_input_names
        }
        with torch.no_grad():
            outputs = model(**inputs)
            # Используем функцию pool для извлечения CLS-токена
            embeddings = pool(
                outputs.last_hidden_state,
                batch["attention_mask"],
                pooling_method="cls",  # Используем CLS-токен
            )
            # Возвращаем CLS-токен (уже имеет размерность [batch_size, hidden_size])
            return {"cls_hidden_state": embeddings.cpu().numpy()}

    cls_dataset = df.map(extract_hidden_states, batched=True, batch_size=128)
    cls_dataset.set_format(type="pandas")
    return pd.DataFrame(
        cls_dataset["cls_hidden_state"].to_list(),
        columns=[f"feature_{n}" for n in range(1, 313)],  # 312 измерений
    )


def compile_dataframe_for_prediction(parsed_data):
    tokenized_df = tokenized_pytorch_tensors(
        parsed_data[[TEXT_FEATURE]],
        column_list=["input_ids", "attention_mask"],
    )

    hidden_states_df = hidden_state_from_text_inputs(tokenized_df)

    NUMERICAL_FEATURE = [
        "positive_reactions_count",
        "negative_reactions_count",
        "message_symbols_count",
        "message_words_count",
        "comment_timedelta_seconds",
        "total_reactions",
        "links_number",
        "emoji_count",
    ]

    CATEGORICAL_FEATURE = [
        "sender_id",
        "is_reply",
        "written_at_night",
        "post_id",
        "year",
        "month",
        "day",
    ]

    preprocessed_df = pd.concat(
        [
            parsed_data[NUMERICAL_FEATURE],
            parsed_data[CATEGORICAL_FEATURE],
            hidden_states_df,
        ],
        axis=1,
    )

    return preprocessed_df


def get_predicts(df):
    return PREDICT_MODEL.predict(df)


async def data_pipeline(group_name, post_id):
    df = await get_comments(group_name, post_id)
    input_df = SinglePostDataframe(data=df)
    input_df = input_df.create_features()

    processed_df = compile_dataframe_for_prediction(input_df)
    preds = pd.Series(get_predicts(processed_df), name="predictions")
    preds = pd.DataFrame(preds)
    output_df = pd.concat([input_df, preds], axis=1)

    return output_df


async def main():
    pass


if __name__ == "__main__":
    asyncio.run(main())
