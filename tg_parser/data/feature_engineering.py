import os
import emoji
import pandas as pd
import ast
import re
import nltk
from nltk import word_tokenize
import string
import numpy as np

# nltk.download('punkt')
# nltk.download('punkt_tab')


class SinglePostDataframe:
    POSITIVE_REACTIONS = [
        ":thumbs_up:",
        ":fire:",
        ":party_popper:",
        ":star_struck:",
        ":red_heart:",
        ":grinning_face_with_smiling_eyes:",
        ":face_with_tears_of_joy:",
        ":smiling_face_with_heart-eyes:",
        ":hundred_points:",
        ":clapping_hands:",
        ":rolling_on_the_floor_laughing:",
    ]

    NEGATIVE_REACTIONS = [
        ":thumbs_down:",
        ":pile_of_poo:",
        ":vomiting_face:",
        ":loudly_crying_face:",
        ":face_with_steam_from_nose:",
        ":angry_face:",
        ":pouting_face:",
        ":face_with_symbols_on_mouth:",
        ":skull:",
    ]

    required_columns = [
        "date",
        "message_id",
        "reply_to_msg_id",
        "channel_name",
        "channel_id",
        "sender",
        "sender_id",
        "message",
        "reactions",
        "channel_members_count",
    ]

    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(
            data, str
        ):  # Для обратной совместимости с путём к файлу
            self.df = pd.read_excel(data, engine="openpyxl")
        else:
            raise ValueError(
                "Неподдерживаемый тип данных. Ожидается DataFrame или путь к файлу."
            )
        self._validate_table()

    def create_features(self):
        self.df["sender"] = self.df["sender"].fillna("Other")
        self.df["sender"] = self.df["sender"].replace(
            "SenderNameBlockByTelegram", "Other"
        )

        self.df["reactions"] = self.df["reactions"].apply(
            self.convert_emoji_to_aliases
        )

        (
            self.df["positive_reactions_count"],
            self.df["negative_reactions_count"],
        ) = zip(*self.df["reactions"].apply(self.count_reactions))

        self.df["total_reactions"] = (
            self.df["positive_reactions_count"]
            + self.df["negative_reactions_count"]
        )

        self.df = self.df.drop(columns=["reactions"])

        self.df["message_symbols_count"], self.df["message_words_count"] = zip(
            *self.df["message"].apply(self.get_message_length)
        )

        self.df["date"] = pd.to_datetime(self.df["date"])

        self.df["comment_timedelta_seconds"] = (
            self.get_timedelta_from_first_comment(self.df["date"])
        )

        self.df["is_reply"] = self.df["reply_to_msg_id"] != int(
            self.return_max_index(self.df)
        )

        self.df = self.df[self.df["sender_id"] != -1]

        self.df["year"] = self.df["date"].dt.year
        self.df["month"] = self.df["date"].dt.month
        self.df["day"] = self.df["date"].dt.day

        self.df["is_reply"] = self.df["is_reply"].astype(int)

        self.df["links_number"] = self.df["message"].apply(self.count_links)

        self.df["emoji_count"] = self.df["message"].apply(self.count_emojis)

        self.df["message"] = self.df["message"].apply(self.process_text_data)

        self.df["written_at_night"] = np.where(
            (self.df["date"].dt.hour >= 0) & (self.df["date"].dt.hour <= 6),
            1,
            0,
        )
        self.df["day_of_week"] = self.df["date"].dt.day_of_week

        self.df["post_id"] = self.df.reply_to_msg_id.value_counts().idxmax()

        self.df = self.df[self.df["message"] != "MediaMessage"]

        self.df["message"] = (
            self.df["message"]
            + "[SEP]"
            + self.df["sender"]
            + "[SEP]"
            + self.df["channel_name"]
        )
        self.df = self.df.drop(
            columns=[
                "date",
                "message_id",
                "reply_to_msg_id",
                "sender",
                "channel_name",
                "day_of_week",
                "channel_id",
                "channel_members_count",
            ]
        )

        self.df = self.df.reset_index()
        self.df = self.df.drop(columns=["index"])

        self.df["sender_id"] = self.df["sender_id"].astype("category")
        self.df["is_reply"] = self.df["is_reply"].astype("category")
        self.df["written_at_night"] = self.df["written_at_night"].astype(
            "category"
        )
        self.df["year"] = self.df["year"].astype("category")
        self.df["month"] = self.df["month"].astype("category")
        self.df["day"] = self.df["day"].astype("category")
        self.df["post_id"] = self.df["post_id"].astype("category")

        return self.df

    def _validate_table(self):
        missing_cols = set(self.required_columns) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Недостающие колонки: {missing_cols}")

    def convert_emoji_to_aliases(self, emoji_dict):
        # Если emoji_dict — это строка, преобразуем её в словарь
        if isinstance(emoji_dict, str):
            try:
                emoji_dict = ast.literal_eval(emoji_dict)
            except (ValueError, SyntaxError) as e:
                print(f"Ошибка при преобразовании строки в словарь: {e}")
                return {}

        # Если emoji_dict — это уже словарь, продолжаем обработку
        alias_dict = {}
        for emoji_char, count in emoji_dict.items():
            # Получаем алиас для эмодзи
            alias = emoji.demojize(emoji_char)
            alias_dict[alias] = count
        return alias_dict

    def count_reactions(self, reaction_dict):
        try:
            if isinstance(reaction_dict, str):
                reaction_dict = ast.literal_eval(reaction_dict)

            positive_count = sum(
                v
                for k, v in reaction_dict.items()
                if k in self.POSITIVE_REACTIONS
            )
            negative_count = sum(
                v
                for k, v in reaction_dict.items()
                if k in self.NEGATIVE_REACTIONS
            )
            return positive_count, negative_count
        except (ValueError, SyntaxError) as e:
            print(f"Ошибка при обработке реакций: {e}")
            return 0, 0

    def get_message_length(self, text):
        return len(text), len(text.split(" "))

    def get_timedelta_from_first_comment(self, column: pd.Series) -> pd.Series:
        """
        Функция принимает на вход столбец датафрейма pandas с датами, обязательно типа datetime64.
        Возвращает новый столбец с разницей в секундах между каждой датой в столбце и временем первого комментария.

        Args:
            column: Столбец датафрейма pandas с типом данных datetime64.

        Returns:
            Новый столбец с разницей в секундах между каждой датой в столбце и временем первого комментария.
        """

        # Проверка типа данных
        if not isinstance(column, pd.Series):
            raise TypeError(
                "Входной аргумент должен быть типом pandas.Series."
            )
        if not pd.api.types.is_datetime64_any_dtype(column):
            raise TypeError("Входной столбец должен быть типом datetime64.")

        # Находим время первого комментария
        first_comment_time = column.min()

        # Вычисляем разницу между каждой датой и временем первого комментария
        return (column - first_comment_time).dt.total_seconds()

    def return_max_index(self, data):
        grouped = data.groupby("reply_to_msg_id").agg({"sender_id": "count"})
        return grouped["sender_id"].idxmax()

    def count_links(self, text):
        # Регулярное выражение для поиска ссылок
        url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )

        # Поиск всех ссылок в тексте
        links = url_pattern.findall(text)

        # Возвращаем количество найденных ссылок
        return len(links)

    def count_emojis(self, text):
        # Регулярное выражение для поиска эмодзи
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # Эмодзи-смайлики
            "\U0001f300-\U0001f5ff"  # Разные символы и пиктограммы
            "\U0001f680-\U0001f6ff"  # Транспорт и карты
            "\U0001f700-\U0001f77f"  # Алхимические символы
            "\U0001f780-\U0001f7ff"  # Геометрические фигуры
            "\U0001f800-\U0001f8ff"  # Дополнительные символы-стрелки
            "\U0001f900-\U0001f9ff"  # Дополнительные символы и пиктограммы
            "\U0001fa00-\U0001fa6f"  # Шахматы
            "\U0001fa70-\U0001faff"  # Символы и пиктограммы
            "\U00002702-\U000027b0"  # Дингбаты
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )

        emojis = emoji_pattern.findall(text)
        return len(emojis)

    def process_text_data(self, text):
        text = re.sub(
            r"\s+", " ", text
        )  # Заменяем последовательности пробелов и других пробельных символов на один пробел
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # Эмодзи-смайлики
            "\U0001f300-\U0001f5ff"  # Разные символы и пиктограммы
            "\U0001f680-\U0001f6ff"  # Транспорт и карты
            "\U0001f700-\U0001f77f"  # Алхимические символы
            "\U0001f780-\U0001f7ff"  # Геометрические фигуры
            "\U0001f800-\U0001f8ff"  # Дополнительные символы-стрелки
            "\U0001f900-\U0001f9ff"  # Дополнительные символы и пиктограммы
            "\U0001fa00-\U0001fa6f"  # Шахматы
            "\U0001fa70-\U0001faff"  # Символы и пиктограммы
            "\U00002702-\U000027b0"  # Дингбаты
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub(r"", text)

        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )  # удаляем ссылки
        text = re.sub(
            r"(?<=[^\w\d])-|-(?=[^\w\d])|[^\w\d\s-]", "", text
        )  # Удаляем все символы, кроме букв, цифр, пробелов и дефисов
        text = re.sub(
            r"\+\d{1,2}\s\(\d{3}\)\s\d{3}-\d{2}-\d{2}", "", text
        )  # удаляем телефонные номера
        text = re.sub(r"\b\w\b", "", text)  # удаляем одиночные символы

        tokens = word_tokenize(text)
        filtered_tokens = [
            token.lower()
            for token in tokens
            if token not in string.punctuation
        ]

        return " ".join(filtered_tokens)


def main():
    df = SinglePostDataframe(
        r"parsed_data/messages_Военный Осведомитель_142234.xlsx"
    )
    df = df.create_features()
    print(df.shape)


if __name__ == "__main__":
    main()
