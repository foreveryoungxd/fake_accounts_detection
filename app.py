import asyncio
import os

import streamlit as st
from dotenv import load_dotenv
from tg_parser.models.inference import data_pipeline, offline_data_pipeline
from tg_parser.utils import save_to_db


load_dotenv()
TABLE_NAME = os.getenv("TABLE_NAME")


# Синхронная обёртка для асинхронной функции
def sync_get_comments(group_username, post_id):
    return asyncio.run(data_pipeline(group_username, post_id))


def split_message(df):
    df = df.copy()
    df["username"] = (
        df["message"]
        .str.split("SEP")
        .str[1]
        .replace(r"[\[\]]", "", regex=True)
    )
    df["channel_name"] = (
        df["message"]
        .str.split("SEP")
        .str[2]
        .replace(r"[\[\]]", "", regex=True)
    )
    df["message"] = (
        df["message"]
        .str.split("SEP")
        .str[0]
        .replace(r"[\[\]]", "", regex=True)
    )

    return df


def process_username_column(df, username_col="username"):
    """
    Обрабатывает колонку с именами пользователей:
    - Если значение равно "Other", заменяет его на "отсутствует".
    - Если значение не равно "Other", заменяет его на ссылку https://t.me/{username}.

    :param df: DataFrame, содержащий колонку с именами пользователей.
    :param username_col: Название колонки с именами пользователей (по умолчанию "username").
    :return: DataFrame с обработанной колонкой.
    """
    # Проверяем, существует ли колонка
    if username_col not in df.columns:
        raise ValueError(f"Колонка '{username_col}' не найдена в DataFrame.")

    # Применяем логику обработки
    df[username_col] = df[username_col].apply(
        lambda x: "отсутствует" if x == "Other" else f"https://t.me/{x}"
    )

    return df


def main():
    # Подключение к базе данных через st.connection
    conn = st.connection("telegram_users_messages_db", type="sql")

    st.title("Детектор дезинформации в комментариях Telegram")
    st.sidebar.title("Выберите режим работы")
    mode = st.sidebar.radio(
        "Режим:",
        options=[
            "Предсказания по ссылке на Telegram-пост",
            "Предсказания по CSV-файлу",
            "Аналитика по базе данных",
        ],
    )

    if mode == "Предсказания по ссылке на Telegram-пост":
        # Ввод данных от пользователя
        group_username = st.text_input(
            "Введите ссылку на канал Telegram (например, https://t.me/milinfolive):"
        )
        post_id = st.number_input(
            "Введите ID поста:", min_value=1, value=1000000
        )

        if st.button("Запустить анализ"):
            # Предсказание
            with st.spinner("Предсказание..."):
                processed_df = sync_get_comments(group_username, post_id)

            processed_df = split_message(processed_df)
            processed_df = processed_df[processed_df["predictions"] == 1]
            processed_df = processed_df[
                ["sender_id", "message", "username", "channel_name", "date"]
            ]
            processed_df = process_username_column(processed_df)

            if len(processed_df) == 0:
                st.write(
                    "Комментарии, содержащие ложную информацию, не обнаружены"
                )
            else:
                st.write(
                    f"Количество комментариев, содержащих ложную информацию: {len(processed_df)}"
                )
                st.dataframe(
                    processed_df.rename(
                        columns={
                            "sender_id": "id пользователя",
                            "message": "комментарий",
                            "username": "ссылка на пользователя",
                            "channel_name": "название канала",
                            "date": "время отправки комментария",
                        }
                    )
                )

                # Сохранение данных в базу данных
                save_to_db(conn, processed_df)

    if mode == "Предсказания по CSV-файлу":
        st.header("Загрузите CSV-файл с комментариями")
        uploaded_file = st.file_uploader(
            "Выберите файл", type=["csv", "xlsx", "xls"]
        )

        if uploaded_file is not None:
            try:
                processed_df = offline_data_pipeline(uploaded_file)
                processed_df = split_message(processed_df)
                processed_df = processed_df[processed_df["predictions"] == 1]
                processed_df = processed_df[
                    [
                        "sender_id",
                        "message",
                        "username",
                        "channel_name",
                        "date",
                    ]
                ]
                processed_df = process_username_column(processed_df)
                st.dataframe(processed_df)

                save_to_db(conn, processed_df)
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {e}")
        else:
            st.info("Пожалуйста, загрузите файл для анализа.")

    if mode == "Аналитика по базе данных":
        st.header("Аналитика по базе данных")
        unique_channels = conn.query(f'select distinct channel_name FROM {TABLE_NAME}')["channel_name"].tolist()

        if unique_channels:
            selected_channel = st.selectbox(
                "Выберите канал для анализа:",
                options=unique_channels,
            )

            if st.button("Получить данные по каналу"):
                query_channel_data = f"""
                        SELECT *
                        FROM telegram_comments
                        WHERE channel_name = '{selected_channel}';
                    """
                channel_data = conn.query(query_channel_data)

                if len(channel_data) == 0:
                    st.write(f"Нет данных для канала '{selected_channel}'.")
                else:
                    st.write(f"Найдено {len(channel_data)} записей для канала '{selected_channel}':")
                    st.dataframe(channel_data)

                    # Статистика
                    st.subheader("Статистика по каналу")
                    st.write(f"Общее количество комментариев: {len(channel_data)}")
                    st.write(f"Первый комментарий: {channel_data['date'].min()}")
                    st.write(f"Последний комментарий: {channel_data['date'].max()}")
        else:
            st.write("В базе данных нет каналов для анализа.")



if __name__ == "__main__":
    main()
