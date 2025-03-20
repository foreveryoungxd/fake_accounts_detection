import asyncio

import streamlit as st

from tg_parser.models.inference import data_pipeline


# Синхронная обёртка для асинхронной функции
def sync_get_comments(group_username, post_id):
    return asyncio.run(data_pipeline(group_username, post_id))


def main():
    st.title("Детектор дезинформации в комментариях Telegram")

    # Ввод данных от пользователя
    group_username = st.text_input(
        "Введите ссылку на канал Telegram (например, https://t.me/milinfolive):"
    )
    post_id = st.number_input("Введите ID поста:", min_value=1, value=1000000)

    if st.button("Запустить анализ"):

        # Предсказание
        with st.spinner("Предсказание..."):
            processed_df = sync_get_comments(group_username, post_id)

        processed_df["username"] = (
            processed_df["message"]
            .str.split("SEP")
            .str[1]
            .replace(r"[\[\]]", "", regex=True)
        )
        processed_df["channel_name"] = (
            processed_df["message"]
            .str.split("SEP")
            .str[2]
            .replace(r"[\[\]]", "", regex=True)
        )
        processed_df["message"] = (
            processed_df["message"]
            .str.split("SEP")
            .str[0]
            .replace(r"[\[\]]", "", regex=True)
        )
        processed_df = processed_df[processed_df["predictions"] == 1]

        st.write(
            f"Количество комментариев, содержащих ложную информацию: {len(processed_df)}"
        )
        st.dataframe(processed_df)
        processed_df.to_excel(
            "test_result.xlsx", index=False, engine="openpyxl"
        )


if __name__ == "__main__":
    main()
