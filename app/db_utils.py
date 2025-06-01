import pymysql
import pandas as pd
import streamlit as st

def get_connection():
    try:
        return pymysql.connect(
            host=st.secrets["DB_HOST"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            database=st.secrets["DB_NAME"],
            port=int(st.secrets["DB_PORT"]),  # Important!
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        raise

## For making Database connection with MYSQL
def get_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        cursorclass=pymysql.cursors.DictCursor
    )


## For Filtering the Employees
def call_filter_employees(role=None, location=None, include_inactive=True):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.callproc("FilterEmployees", [role, location, include_inactive])
            result = cursor.fetchall()
            return pd.DataFrame(result)
    finally:
        connection.close()


## For Adding Experience
def call_experience_bands():
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.callproc("GetEmployeeExperienceBands")
            result = cursor.fetchall()
            return pd.DataFrame(result)
    finally:
        connection.close()


## For simulating the increase
def simulate_global_increment(percent):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.callproc("SimulateGlobalIncrement", [percent])
            result = cursor.fetchall()
            return pd.DataFrame(result)
    finally:
        connection.close()


def get_turnover_rates():
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.callproc("GetTurnoverRates")
            result = cursor.fetchall()
            return pd.DataFrame(result)
    finally:
        connection.close()



def get_industry_benchmarks():
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.callproc("CompareWithIndustry")
            result = cursor.fetchall()
            return pd.DataFrame(result)
    finally:
        connection.close()


def get_employee_rating_diff():
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.callproc("GetEmployeeRatingsDifference")
            result = cursor.fetchall()
            return pd.DataFrame(result)
    finally:
        connection.close()
