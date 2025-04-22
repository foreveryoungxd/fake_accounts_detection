import logging
import os
import re
from collections import Counter
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import text
from telethon.sync import TelegramClient, functions
from telethon.tl.types import ReactionCustomEmoji, ReactionEmoji


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def get_comments(group_username, post_id):
    load_dotenv()
    async with TelegramClient(
        "keepflying",
        os.getenv("api_id"),
        os.getenv("api_hash"),
        device_model="NS685U",
        app_version="5.12.3",
        lang_code="en",
    ) as client:
        try:
            group = await client.get_entity(group_username)
            # Получаем число подписчиков канала
            group_members_count = await client(
                functions.channels.GetFullChannelRequest(group)
            )
            group_members_count = (
                group_members_count.full_chat.participants_count
            )

            logging.info(f"Подключено к группе: {group.title}")
            comments_list = []

            async for comment_message in client.iter_messages(
                group, reply_to=post_id, reverse=True
            ):
                # Получаем текст комментария
                comment_text = comment_message.message.replace("'", '"')
                # Получаем дату и время написания комментария
                comment_date_time = comment_message.date.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                reaction_dict = {}
                if comment_message.reactions:
                    for rc in comment_message.reactions.results:
                        if isinstance(rc.reaction, ReactionEmoji):
                            # Стандартные эмодзи
                            reaction_dict[rc.reaction.emoticon] = rc.count
                        elif isinstance(rc.reaction, ReactionCustomEmoji):
                            # Кастомные эмодзи
                            reaction_dict[
                                f"custom_emoji_{rc.reaction.document_id}"
                            ] = rc.count

                # Получем имя пользователя, оставившего комментарий
                sender = await comment_message.get_sender()
                if sender:
                    if sender.username:
                        sender_name = sender.username
                    else:
                        sender_name = None
                else:
                    sender = None

                # Получаем id сообщения другого пользователя, на который был дан ответ.
                # В ином случае reply_to_msg_id - это id поста от канала
                reply_to_msg_id = comment_message.reply_to.reply_to_msg_id

                print(comment_message.id)
                comments_list.append(
                    {
                        "date": comment_date_time,
                        "message_id": comment_message.id,
                        "reply_to_msg_id": reply_to_msg_id,
                        "channel_name": group.title,
                        "channel_id": group.id,
                        "sender": sender_name,
                        "sender_id": comment_message.sender_id,
                        "message": comment_text,
                        "reactions": reaction_dict,
                        "channel_members_count": group_members_count,
                    }
                )
                # await asyncio.sleep(1)

        except Exception as e:
            comments_list = []
            print(f"Error processing comments: {e}")

        finally:

            await client.disconnect()
            logging.info("Клиент отключен.")

        return pd.DataFrame(comments_list)
        # df.to_excel(f'parsed_data/messages_{group.title}_{post_id}.xlsx', index=False, engine='openpyxl')


def save_to_db(conn, df):
    """
    Сохраняет DataFrame в таблицу базы данных.

    :param conn: Подключение к базе данных через st.connection.
    :param df: DataFrame с данными для сохранения.
    """
    load_dotenv()
    table_name = os.getenv("TABLE_NAME")

    with conn.session as s:
        s.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender_id TEXT,
                    message TEXT,
                    username TEXT,
                    channel_name TEXT,
                    date TEXT,
                    UNIQUE (sender_id, message, date)
                );
                """
            )
        )
        s.commit()

    # Вставляем данные из DataFrame в таблицу
    with conn.session as s:
        for _, row in df.iterrows():
            date_str = (
                row["date"].strftime("%Y-%m-%d %H:%M:%S")
                if pd.notna(row["date"])
                else None
            )
            s.execute(
                text(
                    f"""
                    INSERT OR IGNORE INTO {table_name} (sender_id, message, username, channel_name, date)
                    VALUES (:sender_id, :message, :username, :channel_name, :date);
                    """
                ),
                {
                    "sender_id": row["sender_id"],
                    "message": row["message"],
                    "username": row["username"],
                    "channel_name": row["channel_name"],
                    "date": date_str,
                },
            )
        s.commit()


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


def get_date_range_selector():
    """Создает элементы интерфейса для выбора периода дат"""
    today = datetime.now()
    default_start = today - timedelta(days=7)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Начальная дата", value=default_start, max_value=today
        )
    with col2:
        end_date = st.date_input(
            "Конечная дата", value=today, max_value=today, min_value=start_date
        )

    # Добавляем возможность выбрать "Весь период"
    all_time = st.checkbox("Весь период", value=False)

    if all_time:
        return datetime.now() - timedelta(365), datetime.now()
    return start_date, end_date


def plot_time_analysis(df, time_resolution="H"):
    """Визуализация активности с выделением аномалий"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if time_resolution == "H":
        # Группируем по часам с сохранением даты
        df["time_group"] = df["date"].dt.floor("h")
        title = "Почасовая активность фиктивных аккаунтов"
        xlabel = "Дата и время"

        # Создаем комбинированную метку даты и часа
        time_counts = (
            df.groupby("time_group").size().reset_index(name="counts")
        )
        time_counts["display_label"] = time_counts["time_group"].dt.strftime(
            "%H:%M\n%d.%m"
        )
    else:
        # Группируем по дням
        df["time_group"] = df["date"].dt.date
        title = "Дневная активность фиктивных аккаунтов"
        xlabel = "Дата"
        time_counts = (
            df.groupby("time_group").size().reset_index(name="counts")
        )
        time_counts["display_label"] = time_counts["time_group"].astype(str)

    # Вычисляем аномалии
    if len(time_counts) > 1:
        z_scores = np.abs(stats.zscore(time_counts["counts"]))
        time_counts["is_anomaly"] = z_scores > 1.5
    else:
        time_counts["is_anomaly"] = False

    plt.figure(figsize=(12, 6))

    if time_resolution == "H":
        # Для почасовых данных используем индекс как x-axis
        x = range(len(time_counts))

        # Линейный график
        ax = sns.lineplot(
            x=x,
            y=time_counts["counts"],
            color="royalblue",
            linewidth=2,
            marker="o",
        )

        # Выделение аномальных точек
        anomalies = time_counts[time_counts["is_anomaly"]]
        if not anomalies.empty:
            ax.scatter(
                anomalies.index,
                anomalies["counts"],
                color="red",
                s=100,
                label="Возможная бот-активность",
            )

        # Настраиваем метки оси X
        ax.set_xticks(x)
        ax.set_xticklabels(
            time_counts["display_label"], rotation=45, ha="right"
        )

        # Увеличиваем частоту меток для лучшей читаемости
        if len(time_counts) > 24:  # Если больше суток данных
            ax.set_xticks(x[::6])  # Показываем каждые 6 часов
    else:
        # Для дневных данных
        ax = sns.barplot(
            x="display_label",
            y="counts",
            hue="is_anomaly",
            data=time_counts,
            palette={False: "royalblue", True: "red"},
            dodge=False,
        )
        ax.get_legend().remove()
        plt.xticks(rotation=45, ha="right")

    # Настройка оформления
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Количество комментариев", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    sns.despine(left=True, bottom=True)

    if time_resolution == "H" and not anomalies.empty:
        plt.legend()

    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()


def get_top_users_stats(df, top_n=10):
    """Возвращает статистику по самым активным пользователям"""
    user_stats = (
        df.groupby(["sender_id", "username"])
        .agg(
            total_messages=("message", "count"),
            first_message=("date", "min"),
            last_message=("date", "max"),
        )
        .reset_index()
    )

    user_stats = user_stats.sort_values(
        "total_messages", ascending=False
    ).head(top_n)
    user_stats["activity_period"] = (
        user_stats["last_message"] - user_stats["first_message"]
    ).dt.total_seconds() / 3600
    return user_stats


def preprocess_text(text):
    """Предварительная обработка текста для анализа"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Удаляем пунктуацию
    text = re.sub(r"\d+", "", text)  # Удаляем цифры
    return text


# Список русских стоп-слов
russian_stopwords = [
    "и",
    "в",
    "во",
    "не",
    "что",
    "он",
    "на",
    "я",
    "с",
    "со",
    "как",
    "а",
    "то",
    "все",
    "она",
    "так",
    "его",
    "но",
    "да",
    "ты",
    "к",
    "у",
    "же",
    "вы",
    "за",
    "бы",
    "по",
    "только",
    "ее",
    "мне",
    "было",
    "вот",
    "от",
    "меня",
    "еще",
    "нет",
    "о",
    "из",
    "ему",
    "теперь",
    "когда",
    "даже",
    "ну",
    "вдруг",
    "ли",
    "если",
    "уже",
    "или",
    "ни",
    "быть",
    "был",
    "него",
    "до",
    "вас",
    "нибудь",
    "опять",
    "уж",
    "вам",
    "ведь",
    "там",
    "потом",
    "себя",
    "ничего",
    "ей",
    "может",
    "они",
    "тут",
    "где",
    "есть",
    "надо",
    "ней",
    "для",
    "мы",
    "тебя",
    "их",
    "чем",
    "была",
    "сам",
    "чтоб",
    "без",
    "будто",
    "чего",
    "раз",
    "тоже",
    "себе",
    "под",
    "будет",
    "ж",
    "тогда",
    "кто",
    "этот",
    "того",
    "потому",
    "этого",
    "какой",
    "совсем",
    "ним",
    "здесь",
    "этом",
    "один",
    "почти",
    "мой",
    "тем",
    "чтобы",
    "нее",
    "сейчас",
    "были",
    "куда",
    "зачем",
    "всех",
    "никогда",
    "можно",
    "при",
    "наконец",
    "два",
    "об",
    "другой",
    "хоть",
    "после",
    "над",
    "больше",
    "тот",
    "через",
    "эти",
    "нас",
    "про",
    "всего",
    "них",
    "какая",
    "много",
    "разве",
    "три",
    "эту",
    "моя",
    "впрочем",
    "хорошо",
    "свою",
    "этой",
    "перед",
    "иногда",
    "лучше",
    "чуть",
    "том",
    "нельзя",
    "такой",
    "им",
    "более",
    "всегда",
    "конечно",
    "всю",
    "между",
]


def extract_key_themes(messages, n_clusters=3, n_terms=5):
    """Выделяет ключевые темы из набора сообщений"""
    # Предобработка текстов
    processed_texts = [preprocess_text(msg) for msg in messages]

    # Векторизация текстов
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words=russian_stopwords
    )
    X = vectorizer.fit_transform(processed_texts)

    # Кластеризация сообщений
    if len(messages) >= n_clusters:
        kmeans = KMeans(
            n_clusters=min(n_clusters, len(messages)), random_state=42
        )
        kmeans.fit(X)

        # Собираем ключевые слова для каждого кластера
        terms = vectorizer.get_feature_names_out()
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

        themes = []
        for i in range(kmeans.n_clusters):
            theme_words = [terms[ind] for ind in order_centroids[i, :n_terms]]
            themes.append(", ".join(theme_words))
        return themes
    else:
        # Если сообщений мало, возвращаем самые частые слова
        all_words = " ".join(processed_texts).split()
        word_counts = Counter(all_words)
        return [word[0] for word in word_counts.most_common(n_terms)]
