import asyncio
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from tg_parser.models.inference import data_pipeline, offline_data_pipeline
from tg_parser.utils import (
    extract_key_themes,
    get_date_range_selector,
    get_top_users_stats,
    plot_time_analysis,
    process_username_column,
    save_to_db,
    split_message,
)


load_dotenv()
TABLE_NAME = os.getenv("TABLE_NAME")


# Синхронная обёртка для асинхронной функции
def sync_get_comments(group_username, post_id):
    return asyncio.run(data_pipeline(group_username, post_id))


def main():
    # Подключение к базе данных через st.connection
    conn = st.connection("telegram_users_messages_db", type="sql")

    st.title("Программный модуль выявления ложной информации в Telegram")
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
        post_link = st.text_input(
            "Введите ссылку на новостную публикацию в мессенджере Telegram (например, https://t.me/milinfolive/123456):",
        )

        if not post_link:
            st.write("Вставьте ссылку на telegram-пост")
        else:
            group_username = (
                post_link.rsplit("/", 1)[0] + "/"
            )  # Берем всё до последнего "/" и добавляем "/"
            post_id = post_link.rsplit("/", 1)[1]

            if st.button("Запустить анализ"):
                # Предсказание
                with st.spinner("Предсказание..."):
                    processed_df = sync_get_comments(
                        group_username, int(post_id)
                    )

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
                                "sender_id": "ID пользователя",
                                "message": "Комментарий",
                                "username": "Ссылка на пользователя",
                                "channel_name": "Telegram-канал",
                                "date": "Дата и время",
                            }
                        )
                    )
                    processed_df["date"] = pd.to_datetime(processed_df["date"])
                    plot_time_analysis(processed_df, "H")

                    # Статистика по фиктивным аккаунтам с наибольшей активностью
                    st.subheader("Фиктивные аккаунты с наибольшей активностью")
                    top_users = get_top_users_stats(processed_df)

                    # Форматируем вывод
                    top_users_display = top_users.copy()
                    top_users_display["first_message"] = top_users_display[
                        "first_message"
                    ].dt.strftime("%Y-%m-%d %H:%M")
                    top_users_display["last_message"] = top_users_display[
                        "last_message"
                    ].dt.strftime("%Y-%m-%d %H:%M")
                    top_users_display["activity_period"] = round(
                        top_users_display["activity_period"], 1
                    )

                    st.dataframe(
                        top_users_display.rename(
                            columns={
                                "sender_id": "ID аккаунта",
                                "username": "Ссылка на пользователя",
                                "total_messages": "Всего сообщений",
                                "first_message": "Первое сообщение",
                                "last_message": "Последнее сообщение",
                                "activity_period": "Период активности (часы)",
                            }
                        ).set_index("ID аккаунта")
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

        # Получаем список каналов
        unique_channels = conn.query(
            f"SELECT DISTINCT channel_name FROM {TABLE_NAME}"
        )["channel_name"].tolist()

        if not unique_channels:
            st.warning("В базе данных нет каналов для анализа.")
            return

        # Выбор канала
        selected_channel = st.selectbox(
            "Выберите канал для анализа:",
            options=unique_channels + ["Все каналы"],
        )

        # Выбор периода
        st.subheader("Выберите период анализа")
        start_date, end_date = get_date_range_selector()

        # Выбор детализации для периода
        time_resolution = st.radio(
            "Детализация анализа:",
            options=["По дням", "По часам"],
            horizontal=True,
        )
        resolution_param = "H" if time_resolution == "По часам" else "D"

        if st.button("Загрузить данные"):
            # Формируем базовый запрос
            if selected_channel == "Все каналы":
                query = f"SELECT * FROM {TABLE_NAME} WHERE date BETWEEN '{start_date} 00:00:00' AND '{end_date} 23:59:59'"
            else:
                query = f"SELECT * FROM {TABLE_NAME} WHERE channel_name = '{selected_channel}' AND date BETWEEN '{start_date} 00:00:00' AND '{end_date} 23:59:59'"

            # Выполняем запрос
            channel_data = conn.query(query)

            if len(channel_data) == 0:
                st.warning("Нет данных для выбранного периода и канала.")
                return

            # Преобразуем даты
            channel_data["date"] = pd.to_datetime(channel_data["date"])

            # Основной график
            st.subheader(
                f"Активность фиктивных аккаунтов в период с {start_date} по {end_date}"
            )
            plot_time_analysis(channel_data, resolution_param)

            # Вывод данных
            st.subheader("Выборка данных по запросу")
            st.dataframe(
                channel_data.rename(
                    columns={
                        "sender_id": "ID аккаунта",
                        "message": "Комментарий",
                        "username": "Ссылка на пользователя",
                        "channel_name": "Telegram-канал",
                        "date": "Дата и время",
                    }
                ).set_index("ID аккаунта")
            )

            # Статистика по фиктивным аккаунтам с наибольшей активностью
            st.subheader("Фиктивные аккаунты с наибольшей активностью")
            top_users = get_top_users_stats(channel_data)

            # Форматируем вывод
            top_users_display = top_users.copy()
            top_users_display["first_message"] = top_users_display[
                "first_message"
            ].dt.strftime("%Y-%m-%d %H:%M")
            top_users_display["last_message"] = top_users_display[
                "last_message"
            ].dt.strftime("%Y-%m-%d %H:%M")
            top_users_display["activity_period"] = round(
                top_users_display["activity_period"], 1
            )

            st.dataframe(
                top_users_display.rename(
                    columns={
                        "sender_id": "ID аккаунта",
                        "username": "Ссылка на пользователя",
                        "total_messages": "Всего сообщений",
                        "first_message": "Первое сообщение",
                        "last_message": "Последнее сообщение",
                        "activity_period": "Период активности (часы)",
                    }
                ).set_index("ID аккаунта")
            )

            # Анализ повестки
            st.subheader("Основные темы ложных сообщений")

            # Выбираем только уникальные сообщения для анализа
            unique_messages = channel_data["message"].unique()

            if len(unique_messages) > 0:
                themes = extract_key_themes(unique_messages)

                st.write("**Выявленные ключевые темы:**")
                for i, theme in enumerate(themes, 1):
                    st.write(f"{i}. {theme.capitalize()}")

            else:
                st.warning("Недостаточно сообщений для анализа тем")


if __name__ == "__main__":
    main()
