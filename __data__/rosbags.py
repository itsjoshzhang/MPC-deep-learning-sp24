import os
import sqlite3
from warnings import warn
from dataclasses import dataclass

@dataclass
class RosBagEntry:
    entry_id: int
    topic_name: str
    msg_type: str

    def __init__(self, entry_id, topic_name, msg_type):
        self.entry_id, self.topic_name, self.msg_type = entry_id, topic_name, msg_type

class RosBagConnector:
    def __init__(self, sqlite_file):
        self.__connect(sqlite_file)

    def __enter__(self):
        print(f"Connected to the rosbag. num_messages = {self.countRows('messages')}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__close()
        print("Connection closed. ")

    def __connect(self, sqlite_file):
        """ Make connection to an SQLite database file. """
        self.conn = sqlite3.connect(sqlite_file)
        self.cursor = self.conn.cursor()

    def __close(self):
        """ Close connection to the database. """
        self.conn.close()

    def countRows(self, table_name, print_out=False):
        """ Returns the total number of rows in the database. """
        self.cursor.execute('SELECT COUNT(*) FROM {}'.format(table_name))
        count = self.cursor.fetchall()
        if print_out:
            print('\nTotal rows: {}'.format(count[0][0]))
        return count[0][0]

    def getHeaders(self, table_name, print_out=False):
        """ Returns a list of tuples with column informations:
        (id, name, type, notnull, default_value, primary_key)
        """
        # Get headers from table "table_name"
        self.cursor.execute('PRAGMA TABLE_INFO({})'.format(table_name))
        info = self.cursor.fetchall()
        if print_out:
            print("\nColumn Info:\nID, Name, Type, NotNull, DefaultVal, PrimaryKey")
            for col in info:
                print(col)
        return info

    def getAllElements(self, table_name, print_out=False):
        """ Returns a dictionary with all elements of the table database.
        """
        # Get elements from table "table_name"
        self.cursor.execute('SELECT * from({})'.format(table_name))
        records = self.cursor.fetchall()
        if print_out:
            print("\nAll elements:")
            for row in records:
                print(row)
        return records

    def is_topic(self, topic_name):
        self.cursor.execute(f'SELECT NAME from TOPICS WHERE NAME="{topic_name}"')
        records = self.cursor.fetchall()
        return len(records) > 0 and len(records[0]) > 0

    def getTimeRange(self, topic_name=None):
        command = 'SELECT MIN(TIMESTAMP), MAX(TIMESTAMP) FROM MESSAGES'
        if topic_name is not None:
            topic_id, _ = self.getTopicInfo(topic_name)
            command += f' WHERE TOPIC_ID == {topic_id}'
        self.cursor.execute(command)
        min_t, max_t = self.cursor.fetchall()[0]
        return min_t, max_t

    def getMessageCount(self, topic_name=None):
        command = 'SELECT COUNT(*) FROM MESSAGES'
        if topic_name is not None:
            topic_id, _ = self.getTopicInfo(topic_name)
            command += f' WHERE TOPIC_ID == {topic_id}'
        self.cursor.execute(command)
        return self.cursor.fetchall()[0][0]

    def getTopicInfo(self, topic_name, *, print_out=False):
        if not self.is_topic(topic_name):
            raise KeyError(f"Topic {topic_name} not found! ")
        self.cursor.execute(f'SELECT ID, TYPE FROM TOPICS WHERE NAME="{topic_name}"')
        topic_id, msg_type = self.cursor.fetchall()[0]
        if print_out:
            print(topic_id)
        return topic_id, msg_type

    def getAllMessagesInTopicTimed(self, topic_name, t_min, t_max, *, print_out=False):
        topic_id, msg_type = self.getTopicInfo(topic_name)
        self.cursor.execute(
            f'SELECT TIMESTAMP, DATA FROM MESSAGES '
            f'WHERE TOPIC_ID="{topic_id}" '
            f'AND TIMESTAMP BETWEEN {t_min} AND {t_max}'
        )
        msgs = self.cursor.fetchall()
        if print_out:
            print(f"Found {len(msgs)} data entries.")
        return msgs, msg_type

    def getAllMessagesInTopic(self, topic_name, *, print_out=False, reverse=False, limit=100, offset=100):
        topic_id, msg_type = self.getTopicInfo(topic_name)
        self.cursor.execute(
            f'SELECT TIMESTAMP, DATA FROM MESSAGES '
            f'WHERE TOPIC_ID="{topic_id}" '
            # f'ORDER BY TIMESTAMP DESC '
            f'LIMIT {limit} '
            f'OFFSET {offset}'
        )
        msgs = self.cursor.fetchall()
        if print_out:
            print(f"Found {len(msgs)} data entries.")
        return msgs, msg_type

    def getAllTopicsInfo(self, print_out=False):
        """ Returns a dict of topic-type map.
        """
        # Get all records for 'topics'
        records = self.getAllElements('topics', print_out=False)
        topics_info = {row[1]: row[2] for row in records}

        # Save all topics names
        if print_out:
            print(topics_info)

        return topics_info

    """
    def _getAllMessagesInTopic(self, topic_name, print_out=False):
        "" Returns all timestamps and messages at that topic.
        There is no deserialization for the BLOB data.
        ""
        warn_deprecation(substitute='getAllMesssageInTopic')
        count = 0
        timestamps = []
        messages = []

        # Find if topic exists and its id
        topicFound = self.isTopic(topic_name, print_out=False)

        # If not find return empty
        if not topicFound:
            print('Topic', topic_name, 'could not be found. \n')
        else:
            records = self.getAllElements('messages', print_out=False)

            # Look for message with the same id from the topic
            for row in records:
                if row[1] == topicFound[0]:  # 1 and 0 is 'topic_id'
                    count = count + 1  # count messages for this topic
                    timestamps.append(row[2])  # 2 is for timestamp
                    messages.append(row[3])  # 3 is for all messages

            if print_out:
                print('\nThere are ', count, 'messages in ', topicFound[1])

        return timestamps, messages
    
    def isTopic(self, topic_name, print_out=False):
        "" Returns topic_name header if it exists. If it doesn't, returns empty.
            It returns the last topic found with this name.
        ""
        warn_deprecation(substitute='is_topic')
        boolIsTopic = False
        topicFound = []

        # Get all records for 'topics'
        records = self.getAllElements('topics', print_out=False)

        # Look for specific 'topic_name' in 'records'
        for row in records:
            if row[1] == topic_name:  # 1 is 'name'
                boolIsTopic = True
                topicFound = row
        if print_out:
            if boolIsTopic:
                # 1 is 'name', 0 is 'id'
                print('\nTopic named', topicFound[1], ' exists at id ', topicFound[0], '\n')
            else:
                print('\nTopic', topic_name, 'could not be found. \n')

        return topicFound

    def getAllTopicNames(self, print_out=False):
        "" Returns all topic names.""
        warn_deprecation(substitute='getAllTopicsInfo')
        topicNames = []
        # Get all records for 'topics'
        records = self.getAllElements('topics', print_out=False)

        # Save all message types
        for row in records:
            topicNames.append(row[1])  # 1 is for topic name
        if print_out:
            print('\nMessages types are:')
            print(topicNames)

        return topicNames

    def getAllMsgsTypes(self, print_out=False):
        "" Returns all messages types.
        ""
        warn_deprecation('getAllTopicsInfo')
        msgsTypes = []
        # Get all records for 'topics'
        records = self.getAllElements('topics', print_out=False)

        # Save all message types
        for row in records:
            msgsTypes.append(row[2])  # 2 is for message type
        if print_out:
            print('\nMessages types are:')
            print(msgsTypes)

        return msgsTypes

    def getMsgType(self, topic_name, print_out=False):
        "" Returns the message type of that specific topic.
        ""
        warn("This method is deprecated. Use getAllTopicsInfo instead. ")
        msg_type = []
        # Get all topics names and all message types
        topic_names = self.getAllTopicsNames(print_out=False)
        msgs_types = self.getAllMsgsTypes(print_out=False)

        # look for topic at the topic_names list, and find its index
        for index, element in enumerate(topic_names):
            if element == topic_name:
                msg_type = msgs_types[index]
        if print_out:
            print('\nMessage type in', topic_name, 'is', msg_type)

        return msg_type
    """

def warn_deprecation(substitute):
    warn(f'This method is deprecated. Use {substitute} instead.')

if __name__ == '__main__':
    files = [f for f in os.listdir("rosbags") if not f.startswith(".")]
    topic, i = "", 0
    for file in sorted(files):
        print(f"\n{file}")
        
        with RosBagConnector(f"rosbags/{file}/{file}_GPS_0.db3") as bag:
            if i == 0:
                (topic,_), = bag.getAllTopicsInfo(print_out=True).items()
            msgs,_ = bag.getAllMessagesInTopic(topic, print_out=True, limit=10000)
            i += 1