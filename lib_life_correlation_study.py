import json
import os
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import xml.etree.ElementTree as ET
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import numpy as np

def extract_canvas_json(filename, target_author):    
    with open(filename, "r") as file:
        data = json.load(file)
    
    records = []
    for course in data:
        for assignment in course["assignments"]:
            course_id_match = re.search(r"/courses/(\d+)/", assignment["html_url"])
            course_id = int(course_id_match.group(1)) if course_id_match else None
            
            for submission in assignment["submissions"]:
                raw_score = submission["raw_score"]
                total_possible_points = submission["total_possible_points"]
                
                records.append({
                    "id": int(assignment["id"]),
                    "course_id": course_id,
                    "course_name": None,
                    "type": "assignemnt",
                    "title": assignment["title"],                    
                    "raw_score": float(raw_score) if raw_score and raw_score.replace('.', '', 1).isdigit() else 0.0,
                    "total_possible_points": float(total_possible_points) if total_possible_points and total_possible_points.replace('.', '', 1).isdigit() else 0.0,
                    "assigned_date": datetime.strptime(assignment["assigned_date"], "%B %d, %Y %I:%M %p") if assignment["assigned_date"] else None,
                    "due_date": datetime.strptime(assignment["due_date"], "%B %d, %Y %I:%M %p") if assignment["due_date"] else None,
                    "source": "canvas"
                })
    
    for course in data:
        for discussion in course.get("discussions", []):
            course_id_match = re.search(r"/courses/(\d+)/", discussion["url"])
            course_id = int(course_id_match.group(1)) if course_id_match else None
            
            for topic_entry in discussion.get("topic_entries", []):
                if topic_entry["author"] == target_author:
                    posted_date = datetime.strptime(topic_entry["posted_date"], "%B %d, %Y %I:%M %p")
                    records.append({
                        "id": int(discussion["id"]),
                        "course_id": course_id,
                        "course_name": None,
                        "type": "discussion",
                        "title": discussion["title"],
                        "due_date": posted_date,
                        "source": "canvas"
                    })

    # Convert do df
    df = pd.DataFrame(records)
    
    # Add cognitive load
    df = add_cognitive_load(df)
    
    # Map course name and id
    df = course_id_name_mapper(df)
    
    return df  

def extract_year_from_filename(filename):
    match = re.search(r'_(\d{4})_', filename)
    return int(match.group(1)) if match else datetime.now().year

def extract_course_name_from_filename(filename):
    match = re.match(r'([^_]+)_\d{4}_.*', filename)
    return match.group(1) if match else None

def extract_gradescope_CSVs(folder_path):
    all_data = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            course_year = extract_year_from_filename(file)
            course_name = extract_course_name_from_filename(file)
            df = pd.read_csv(file_path)
            
            # Rename columns
            df = df.rename(columns={
                "Name": "title",
                "Points": "raw_score",
                "Total Points": "total_possible_points",
                "Released": "assigned_date",
                "Due Date": "due_date"
            })
            
            # Verify data types
            df["raw_score"] = pd.to_numeric(df["raw_score"], errors='coerce').fillna(0.0)
            df["total_possible_points"] = pd.to_numeric(df["total_possible_points"], errors='coerce').fillna(0.0)
            
            # Parse dates with extracted year
            def parse_date(date_str):
                if pd.isna(date_str):
                    return None
                try:
                    return datetime.strptime(f"{date_str}, {course_year}", "%b %d at %I:%M%p, %Y")
                except ValueError:
                    return None
            
            df["assigned_date"] = df["assigned_date"].apply(parse_date)
            df["due_date"] = df["due_date"].apply(parse_date)
            
            # Default type value
            df["type"] = "assignment"
            df["source"] = "gradescope"
            df["course_name"] = course_name
            df["course_id"] = None
            
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Add cognitive load
        combined_df = add_cognitive_load(combined_df)
        
        # Map course name and id
        df = course_id_name_mapper(df)
        
        return combined_df
    else:
        return pd.DataFrame()

def preprocess_adjust_dates(df):
    # Define thresholds - Subjectively chosen
    default_thresholds = {"default": 3, "test": 4, "quiz": 2, "project": 14}
    max_thresholds = {"default": 7, "test": 7, "quiz": 5, "project": 21}

    def adjust_row(row):
        category = str(row["task_category"]).lower()
        default_days = default_thresholds.get(category, default_thresholds["default"])
        max_days = max_thresholds.get(category, max_thresholds["default"])

        if pd.isna(row["assigned_date"]) and pd.isna(row["due_date"]):
            return row 

        if pd.isna(row["assigned_date"]):
            row["assigned_date"] = row["due_date"] - pd.Timedelta(days=default_days)

        if pd.isna(row["due_date"]):
            row["due_date"] = row["assigned_date"] + pd.Timedelta(days=default_days)

        # Enforce max threshold
        if (row["due_date"] - row["assigned_date"]).days > max_days:
            row["assigned_date"] = row["due_date"] - pd.Timedelta(days=max_days)

        return row

    return df.apply(adjust_row, axis=1)

def task_load_per_day(df):
    # Entire date range
    all_dates = pd.date_range(df['assigned_date'].min(), df['due_date'].max())

    # Count occurrences per each day
    date_counts = {date: ((df['assigned_date'] <= date) & (df['due_date'] >= date)).sum() for date in all_dates}

    # Convert to df
    result_df = pd.DataFrame(list(date_counts.items()), columns=['date', 'count'])

    return result_df

def plot_work_intervals(df, title='Intervals of Assigned and Due Dates by Course'):
    fig = go.Figure()

    # Unique colors per course
    unique_courses = df['course_name'].unique()
    color_map = {course: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, course in enumerate(unique_courses)}

    # Track added legends
    added_legends = set()

    # Add interval lines with course-specific colors
    for i, row in df.iterrows():
        course_name = row['course_name']
        show_legend = course_name not in added_legends
        added_legends.add(course_name)

        fig.add_trace(go.Scatter(
            x=[row['assigned_date'], row['due_date']],
            y=[i, i],
            mode='lines+markers',
            marker=dict(size=8, color=color_map[course_name]),
            line=dict(color=color_map[course_name], width=3),
            name=course_name,
            legendgroup=course_name,    # Group legend entries
            showlegend=show_legend,     # Show legend only once per course
            hoverinfo='text',
            text=f'ID: {row.get("id", 0)}<br>'
                 f'Title: {row.get("title", "Unknown")}<br>'
                 f'Course: {course_name}<br>'
                 f'Start: {row["assigned_date"].date()}<br>'
                 f'End: {row["due_date"].date()}'
        ))

    # Formatting
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis=dict(showticklabels=False),  # Remove y-axis labels
        hovermode='x',
        template='plotly_white',
        showlegend=True
    )

    fig.show()

# Arbitrary rating based on the perceived difficulty of the assignemnt.
def add_cognitive_load(df):
    def assign_points(title):
        title = title.lower()
        if 'test' in title:
            return 10
        elif 'quiz' in title:
            return 7
        elif 'project' in title:
            return 15
        else:
            return 3
    
    df['cognitive_load'] = df['title'].apply(assign_points)
    
    def categorize_task(title):
        title = title.lower()
        if 'test' in title:
            return 'test'
        elif 'quiz' in title:
            return 'quiz'
        elif 'project' in title:
            return 'project'
        else:
            return 'default'
    
    df['task_category'] = df['title'].apply(categorize_task)
    
    return df

def compute_stress(df, start_date, end_date, alpha=1):
    date_range = pd.date_range(start=start_date, end=end_date)
    stress_values = []
    
    for t in date_range:
        stress_t = 0
        overlapping_tasks = 0
        
        for _, row in df.iterrows():
            assigned_date, due_date = row['assigned_date'], row['due_date']
            w = row['cognitive_load']               # Task weight based on effort
            D = max((row['due_date'] - t).days, 0)  # Days until due date
            
            if assigned_date <= t <= due_date:
                stress_t += w / (D + 1)
                overlapping_tasks += 1
        
        stress_t += alpha * overlapping_tasks
        stress_values.append({'date': t, 'stress': stress_t})
    
    return pd.DataFrame(stress_values)

def plot_stress(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df['stress'], marker='o', linestyle='-', color='b', label='Stress Level')
    plt.xlabel('Date')
    plt.ylabel('Stress Level')
    plt.title('Stress Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_task_load(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df['count'], marker='o', linestyle='-', color='b', label='Active Assignments')
    plt.xlabel('Date')
    plt.ylabel('# Active Assignments')
    plt.title('# of Active Assignments Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.show()

def plot_stress_measures(stress_df, task_df, title="Stress and Active Assignments Over Time"):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot task count (background)
    ax2 = ax1.twinx()
    ax2.plot(task_df['date'], task_df['count'], marker='o', linestyle='-', color='r', label='Active Assignments', alpha=0.3)
    ax2.set_ylabel('# Active Assignments', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Plot stress levels (foreground)
    ax1.plot(stress_df['date'], stress_df['stress'], marker='o', linestyle='-', color='b', label='Stress Level')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stress Level', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Titles and grid
    plt.title(title)
    fig.autofmt_xdate()
    ax1.grid()

    # Legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()

# Manual mapping of course ID to course name for Gradescope Merge
def course_id_name_mapper(df):
    course_map = {
        78480: "ECE4501 - AI Hardware",
        74998: "CS2910",
        102343: "CS4501 - AI-Powered Cybersecurity",
        118081: "ECE6501 - AMR",
        116798: "ECE4991",
        117327: "STS4500",
        118128: "KLPA1400",
        94905: "ECE4435",
        94366: "ECE4750",
        131872: "STS4600",
        133239: "CS6888",
        131801: "KLPA1400 - S25",
        106115: "ECE6501 - Advanced Embedded Systems",
        131814: "CS6501 - Analyzing Online Behavior",
        79663: "CS3130",
        74814: "CS3240",
        130855: "CS6501 - Data Privacy",
        93778: "CS3100",
        59093: "CS3710",
        132957: "CS 6501 - Network Security"
    }
    
    # Ensure course_id is an integer for mapping
    df["course_id"] = pd.to_numeric(df["course_id"], errors='coerce')
    
    # Fill missing course_name based on course_id
    df["course_name"] = df.apply(lambda row: course_map.get(row["course_id"], row["course_name"]), axis=1)
    
    # Fill missing course_id based on course_name
    reverse_map = {v: k for k, v in course_map.items()}
    df["course_id"] = df.apply(lambda row: reverse_map.get(row["course_name"], row["course_id"]), axis=1)
    
    return df

def get_stress_measures(df):
    # Task load per day
    df_work_per_day = task_load_per_day(df)
    
    ## Validate Task Load Test
    # df_work_per_day_sorted = df_work_per_day.sort_values(by='count', ascending=False)
    # print(f'Most assignments per day \n{df_work_per_day_sorted.head(10)}')
    
    #  Multi-task load of weighted assignments with approaching deadlines.
    df_stress = compute_stress(df, df['assigned_date'].min(), df['due_date'].max())

    return df_work_per_day, df_stress

def get_sleep_hours(df):
    # Convert 'Day-Time' to datetime
    df["Day-Time"] = pd.to_datetime(df["Day-Time"])

    finish_indices = df[df["FinishDay"] == 1].index
    start_indices = df[df["StartDay"] == 1].index

    sleep_data = []

    for finish_idx in finish_indices:
        next_start_idx = start_indices[start_indices > finish_idx].min()
        if pd.notna(next_start_idx):
            sleep_duration = (df.loc[next_start_idx, "Day-Time"] - df.loc[finish_idx, "Day-Time"]).total_seconds() / 3600  # Convert to hours
            sleep_data.append({"LogicalDate": df.loc[finish_idx, "LogicalDate"], "time_slept": sleep_duration})

    # Create df
    sleep_df = pd.DataFrame(sleep_data)

    return sleep_df

def plot_sleep_hours(df):
    plt.figure(figsize=(10, 5))
    plt.scatter(df["LogicalDate"], df["time_slept"], marker='o', linestyle='-', color='b', label='Time Slept (hours)')
    plt.xlabel("Date")
    plt.ylabel("Time Slept (hours)")
    plt.title("Time Slept Over Time")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.show()

def extract_fitness_xml(filename):
    try:
        # Parse XML data
        tree = ET.parse(filename)
        root = tree.getroot()

        data = []
        for record in root.findall("Record"):
            start_date_str = record.get("startDate")
            end_date_str = record.get("endDate")

            # Convert date strings to datetime objects
            start_date = datetime.strptime(start_date_str.split(" -")[0], "%Y-%m-%d %H:%M:%S")
            end_date = datetime.strptime(end_date_str.split(" -")[0], "%Y-%m-%d %H:%M:%S")

            formatted_start_date = start_date.strftime("%B %d, %Y %I:%M %p")
            formatted_end_date = end_date.strftime("%B %d, %Y %I:%M %p")

            data.append({
                "Record type": record.get("type"),
                "unit": record.get("unit"),
                "startDate": formatted_start_date,
                "endDate": formatted_end_date,
                "value": record.get("value"),
            })

        # Create and return df
        return pd.DataFrame(data)

    except ET.ParseError:
        print("Error: Unable to parse the XML file. Please check its format.")
        return pd.DataFrame()

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return pd.DataFrame()

    except Exception as e:
        print(f"Unexpected error: {e}")
        return pd.DataFrame()

def fitness_measure_per_day(df, record_name="HKQuantityTypeIdentifierStepCount",measure_label="step count"):
    # Filter only measure count records
    df = df[df["Record type"] == record_name].copy()

    # Convert 'startDate' to datetime
    df["date"] = pd.to_datetime(df["startDate"], format="%B %d, %Y %I:%M %p").dt.date

    # Convert 'value' to numeric
    df["value"] = pd.to_numeric(df["value"])

    # Group by date and sum the measure counts
    daily_measure = df.groupby("date")["value"].sum().reset_index()

    # Rename columns
    daily_measure.columns = ["date", measure_label]

    return daily_measure

def plot_scatter_per_date(df, value_col, title="", ylabel=""):
    df["date"] = pd.to_datetime(df["date"])

    # Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(df["date"], df[value_col], color="blue", alpha=0.6)

    # Formatting
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Show
    plt.show()

# Plot two value columns
def plot_scatter_compare(df, x_col, y_col, title="Scatter Plot with Trend Line"):
    # Ensure numerical columns are numeric
    df = df[[x_col, y_col]].dropna()  # Drop missing values
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    # Sort data
    df = df.sort_values(by=x_col)

    # Scatter plot
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=df[x_col], y=df[y_col], color="blue", alpha=0.6, label="Data Points")

    # Compute and plot trend line (linear regression)
    m, b = np.polyfit(df[x_col], df[y_col], 1)  # Fit line: y = mx + b
    plt.plot(df[x_col], m * df[x_col] + b, color="red", linestyle="--", label="Linear Regression")

    # Formatting
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Show the plot
    plt.show()

def extract_gaming_excel(filename):
    # Load XLSX
    df = pd.read_excel(filename)

    # Select required columns
    selected_columns = ["match_id", "win", "game_creation_date", "duration"]
    df_selected = df[selected_columns].copy()

    # Convert game_creation_date to datetime
    df_selected["game_creation_date"] = pd.to_datetime(df_selected["game_creation_date"], errors="coerce")

    # Rename column
    df_selected.rename(columns={"duration": "duration_sec"}, inplace=True)

    return df_selected

def get_gaming_time_per_day(df):
    # Ensure datetime format
    df["game_creation_date"] = pd.to_datetime(df["game_creation_date"], errors="coerce")

    # Extract only date
    df["game_date"] = df["game_creation_date"].dt.date

    # Group by date and sum duration
    gaming_time_per_day = df.groupby("game_date")["duration_sec"].sum().reset_index()

    # Rename columns
    gaming_time_per_day.rename(columns={"duration_sec": "total_gaming_time_sec", "game_date": "date"}, inplace=True)

    # Convert to hours
    gaming_time_per_day["total_gaming_time_hours"] = gaming_time_per_day["total_gaming_time_sec"] / 3600

    return gaming_time_per_day