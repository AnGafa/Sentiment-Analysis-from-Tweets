import mysql.connector
from mysql.connector import Error
import config

try:

    connection = mysql.connector.connect(host=config.host,
                                        port=config.port,
                                        user=config.user,
                                        password=config.password)
    if connection.is_connected():
        mycursor = connection.cursor()
        mycursor.execute("CREATE DATABASE twitterScrape")
        print("Database created successfully")
        mycursor.close()
        connection.close()

        connection = mysql.connector.connect(host=config.host,
                                        port=config.port,
                                        user=config.user,
                                        password=config.password,
                                        database = 'twitterScrape')
        mycursor = connection.cursor()
        mycursor.execute("CREATE TABLE tweets (id INT AUTO_INCREMENT PRIMARY KEY, created_at DATETIME, text VARCHAR(255), location VARCHAR(255))")
        print("Table created successfully")
        mycursor.close()
        connection.close()
        print("MySQL connection is closed")

except Error as e:
    print("Error while connecting to MySQL", e)