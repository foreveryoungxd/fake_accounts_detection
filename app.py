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
    post_id = st.number_input("Введите ID поста:", min_value=1, value=142234)

    if st.button("Запустить анализ"):

        # Предсказание
        with st.spinner("Предсказание..."):
            processed_df = sync_get_comments(group_username, post_id)

        # Отображение результатов
        st.write("Результаты:")
        st.write(f"Количество комментариев: {len(processed_df)}")
        st.table(processed_df)
        # st.write(f"Количество дезинформационных комментариев: {sum(preds)}")


if __name__ == "__main__":
    main()
