# 连接数据库的操作  
import sys
import os

# 将当前文件所在目录添加到Python路径
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import os
import uuid
import shutil
import re
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime
from config import MYSQL_CONFIG

# 配置参数

#db_connection =create_db_connection(),建立数据库链接
class kgSQL():
    def __init__(self,database='',env=''):   # 为空就是默认的库
        # 初始化数据库连接
        self.env=env.strip()
        self.database=database.strip()
        self.db_connection = self.create_db_connection()
        
        
    def create_db_connection(self,):
        """创建数据库连接"""
        dbase_name=self.database if self.database else MYSQL_CONFIG['database']
        '''if self.env =="dev":
            dbase_name=f"{MYSQL_CONFIG['database']}_dev"'''

        try:
            connection = mysql.connector.connect(
                host=MYSQL_CONFIG['host'],
                database=dbase_name, 
                user=MYSQL_CONFIG['user'],
                password=MYSQL_CONFIG['password'],
                port=MYSQL_CONFIG['port']
            )
            if connection.is_connected():
                db_info = connection.get_server_info()
                print(f"成功连接到MySQL服务器: {db_info}")
                return connection
        except Error as e:
            print(f"数据库连接错误: {e}")
            raise

    def _ensure_connected(self):
        """确保数据库连接有效，无效则重新连接,2025 -10 -11,解决mysql连接失败的问题"""
        try:
            if not self.db_connection or not self.db_connection.is_connected():
                print("数据库连接已断开，尝试重新连接...")
                self.db_connection = self.create_db_connection()
            else:
                cursor = None
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT 1")

                if cursor:
                    cursor.close()
        except Exception as e:
            print(f"连接验证失败，重新连接: {e}")
            self.db_connection = self.create_db_connection()

    def execute_query(self,query, params=None, commit=False):
        """单次执行SQL语句"""
        #db_connection =create_db_connection()
        # 确保连接有效
        self._ensure_connected()
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute(query, params or ())
            if commit:  # 插入模式/更新模式
                self.db_connection.commit()
                # 对于UPDATE语句，返回受影响的行数；对于INSERT语句，返回lastrowid
                if query.strip().upper().startswith('UPDATE'):
                    return cursor.rowcount
                else:
                    return cursor.lastrowid
                
            else:  # 查询模式
                print("进行数据查询======")
                return cursor.fetchall()  # 查询为空如何？
        except Error as e:
            print(f"执行查询错误: {e}")
            if commit:
                self.db_connection.rollback()
            # 发生错误时尝试重新连接，为下一次操作做准备
            self.db_connection = self.create_db_connection()
            raise
        finally: #cursor.close()
            if cursor:
                cursor.close()  # 确保游标关闭
            print("sql使用完毕！")

    def batch_insert_query(self,query, datalist, commit=False):
        """批量执行SQL语句"""
        #db_connection =create_db_connection()
        # 确保连接有效
        self._ensure_connected()
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            #print("批量开始了sqlmanager=================")
            cursor.executemany(query, datalist)
            
            if commit:  # 插入/更新/删除模式
                self.db_connection.commit()
                #print("批量开始了sqlmanager================")
                return cursor.rowcount  # 返回受影响的行数（批量插入数量）#return cursor.lastrowid
            else:  # 查询模式
                print("进行数据查询======")
                return cursor.fetchall()  # 查询为空如何？
        except Error as e:
            print(f"执行查询错误: {e}")
            if commit:
                self.db_connection.rollback()
            # 发生错误时尝试重新连接，为下一次操作做准备
            self.db_connection = self.create_db_connection()
            raise
        finally: #cursor.close()
            if cursor:
                cursor.close()  # 确保游标关闭
            print("sql使用完毕！")

    def get_last_insert_id(self) -> int:
        """
        获取最后插入的自增ID
        Returns:最后插入的自增ID
        """
        result = self.execute_query("SELECT LAST_INSERT_ID() as last_id")
        if result and len(result) > 0:
            return result[0]['last_id']
        return 0
            
    def add(self,):
        return

    def delete(self,):
        return
    def update(self,):
        return

    def close(self):
        """关闭连接"""
        if self.db_connection and self.db_connection.is_connected():
            self.db_connection.close()
            print("数据库连接已关闭")

def insert_task_node(mysql,data):
    # 插入数据到任务节点 数据表里面:有节点英文名，就能统计一个任务下节点的次数
    sql=f"""
    INSERT INTO task_node_info (create_time, use_id, task_id,node_name_en, output, end_time, cost_time)
    VALUES (%s, %s, %s, %s, %s, %s,%s)
    """
    if isinstance(data,dict):
        print("2单条数据插入")
        params = (data['create_time'], data['use_id'], data['task_id'],data['node_name_en'], data['output'],data['end_time'],data['cost_time'])
        res_id=mysql.execute_query(sql, params, commit=True)
        return res_id
    else:
        print("insert_task_node插入数据异常，插入失败！",type(data))
        return 
def insert_task(mysql,data):
    # 插入数据到数据表里面
    sql=f"""
    INSERT INTO task_info (create_time, use_id, flag, file_name_list, file_minio_list, end_time)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    if isinstance(data,dict):
        print("1单条数据插入")
        params = (data['create_time'], data['use_id'], data['flag'], data['file_name_list'],data['file_minio_list'],data['end_time'])
        res_id=mysql.execute_query(sql, params, commit=True)
        return res_id
    else:
        print("insert_task插入数据异常，插入失败！",type(data))
        return -1

def insert_node_base(mysql,data):
    # 插入数据到数据表里面
    sql=f"""
    INSERT INTO node_base_info (create_time, name_zh, name_en, description)
    VALUES (%s, %s, %s, %s)
    """
    if isinstance(data,dict):
        print("单条数据插入")
        params = (data['create_time'], data['name_zh'], data['name_en'], data['description'])
        res_id=mysql.execute_query(sql, params, commit=True)
        return res_id
    else:
        print("insert_node_base插入数据异常，插入失败！",type(data))
        return 

def insert_conversation(mysql,data):
    # 插入数据到数据表里面info=首次的问题，字符串
    sql=f"""
    INSERT INTO conversation_info (id, create_time, use_id, task_id, info,session_id,type)
    VALUES (%s, %s, %s, %s, %s,%s,%s)
    """
    #id=str(uuid.uuid4())
    if isinstance(data,dict):
        print("单条数据插入")
        params = (data['id'],data['create_time'], data['use_id'], data['task_id'],data['info'],data['session_id'],data['type'])
        res_id=mysql.execute_query(sql, params, commit=True)
        return res_id
    else:
        print("insert_conversation插入数据异常，插入失败！",type(data))
        return 

def insert_message(mysql,data):
    # 插入数据到数据表里面,info=一个问答对json字符串{q:'',a:''} conversation_id
    sql=f"""
    INSERT INTO message_info (id, create_time, doc_id, task_id, info,type)
    VALUES (%s, %s, %s, %s, %s,%s)
    """
    #id=str(uuid.uuid4())
    if isinstance(data,dict):
        print("单条数据插入")
        if isinstance(data['info'],dict):
            info=json.dumps(data['info'],ensure_ascii=False)
        else:
            info=str(data['info'])
        params = (data['id'],data['create_time'], data['doc_id'], data['task_id'], info,data['type']) #conversation
        res_id=mysql.execute_query(sql, params, commit=True)
        return res_id
    else:
        print("insert_message插入数据异常，插入失败！",type(data))
        return

#支持按任意条件查询 task_node_info信息
def search_task_node(mysql, query_conditions=None, params=None):
    """
    查询 task_node_info 表，主要获取 output 字段内容
    :param mysql: kgSQL 实例
    :param query_conditions: 查询条件，如 "task_id = %s AND node_name_en = %s"
    :param params: 条件参数，元组如 (123, "extract_node")
    :return: 查询结果列表 [{'output':'xxx'}, ...]
    """
    # 基础 SQL
    sql = "SELECT id, output, node_name_en, task_id, create_time, end_time, cost_time FROM task_node_info"
    
    # 如果有查询条件，拼接 WHERE
    if query_conditions:
        sql += " WHERE " + query_conditions
    # 执行查询
    result = mysql.execute_query(sql, params=params, commit=False)
    return result
# 只传 task_node_id 就获取 output
def get_output_by_id(mysql, task_node_id):
    """根据 task_id 获取所有节点的 output"""
    sql = "SELECT node_name_en, output FROM task_node_info WHERE id = %s"
    result=mysql.execute_query(sql, (task_node_id,),commit=False)
    return result

if __name__=="__main__":
    print("ok")
    # 1. 创建数据库连接
    db = kgSQL()
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 一条完整的 task_node_info 测试数据
    test_node_data = {
        "create_time":now_time,
        "use_id": "user_1001",          # 用户ID
        "task_id": 22,         # 任务ID（数字）
        "node_name_en": "doc_parse",     # 节点英文名
        "output": "文档解析完成，共提取文本 1280 字",  # 输出结果
        "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cost_time": 5.2                 # 耗时（秒）
    }
    # 2. 插入这条数据
    insert_id = insert_task_node(db, test_node_data)
    print(f"插入成功，返回ID：{insert_id}")

    # 3. 关闭连接
    db.close()
    # ======================
    # 示例 1：按 task_id 查询，多条
    # ======================
    '''task_id = 1
    node_id=1
    res = search_task_node(mysql=db, query_conditions="task_id = %s",params=(task_id,))
    print("按 task_id 查询结果：")
    for item in res:
        print(f"节点：{item['node_name_en']}，输出：{item['output']}")

    # ======================
    # 示例 2：按 task_id + 节点英文名 查询,只有一条/多条，重复执行一个节点
    # ======================
    res2 = search_task_node(mysql=db,query_conditions="task_id = %s AND node_name_en = %s",params=(task_id, "test",))
    if res2:
        print("最终输出结果：", res2[0]['output'])
    # ======================
    # 示例 3：按 task_node_id 查询，1条
    # ======================
    res3 = get_output_by_id(db, node_id)
    print("查询结果：", res3)

    # 关闭连接
    db.close()'''