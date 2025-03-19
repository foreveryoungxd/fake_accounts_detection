import asyncio
import logging
import os

import pandas as pd
from dotenv import load_dotenv
from telethon.sync import TelegramClient, functions


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def get_comments(group_username, post_id):
    load_dotenv()
    async with TelegramClient(
        "keepflying",
        os.getenv("api_id"),
        os.getenv("api_hash"),
        device_model="Swift SF314-43",
        app_version="5.11.1",
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

                # Подсчитываем количество каждой реакции на комментарий
                reaction_dict = {}
                if comment_message.reactions:
                    reaction_dict = {
                        rc.reaction.emoticon: rc.count
                        for rc in comment_message.reactions.results
                    }

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


async def main():
    df = await get_comments("https://t.me/milinfolive", 142234)
    print(df.shape)


if __name__ == "__main__":
    asyncio.run(main())
