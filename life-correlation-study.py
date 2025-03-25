from lib_sleep_study import *
from lib_location_study import *
from lib_life_correlation_study import *

from scipy.stats import pearsonr
import statsmodels.api as sm

if __name__ == "__main__":
    '''
    Work Data
    '''
    ## Process Canvas Data 
    # Extract
    df_canvas = extract_canvas_json("work_data/all_output.json", target_author="Gabriel Gladstone")
    
    # Preprocess
    df_canvas = preprocess_adjust_dates(df_canvas)
    
    # Drop duplicates
    df_canvas = df_canvas.drop_duplicates(subset='id', keep='first')
    
    # Plot
    plot_work_intervals(df_canvas, title='Canvas Intervals of Assigned and Due Dates by Course')
    
    ## Determine Canvas Stress Measures 
    df_canvas_work_per_day, df_canvas_stress = get_stress_measures(df_canvas)
    plot_stress_measures(df_canvas_stress, df_canvas_work_per_day, title="Canvas Stress and Active Assignments Over Time")
    
    ## Process Gradescope Data
    # Extract
    folder_path = "work_data"
    df_gradscope = extract_gradescope_CSVs(folder_path)
    
    # Plot
    plot_work_intervals(df_gradscope, title='Gradescope Intervals of Assigned and Due Dates by Course')
    
    ## Determine Gradescope Stress Measures
    df_gradescope_work_per_day, df_gradescope_stress = get_stress_measures(df_gradscope)
    plot_stress_measures(df_gradescope_stress, df_gradescope_work_per_day, title="Gradescope Stress and Active Assignments Over Time")
    
    ## Combine Work Data 
    df_all_work = pd.concat([df_canvas, df_gradscope], ignore_index=True)
    # Remove duplicate record (keeping gradescope)
    df_all_work = df_all_work.loc[df_all_work.groupby(['course_name', 'title'])['source']
                              .transform(lambda x: x == 'gradescope') | 
                              ~df_all_work.duplicated(subset=['course_name', 'title'], keep=False)]
    
    # Plot
    plot_work_intervals(df_all_work, title='All Work Intervals of Assigned and Due Dates by Course')
    
    ## Determine Combine Stress Measures
    df_all_work_work_per_day, df_all_work_stress = get_stress_measures(df_all_work)
    plot_stress_measures(df_all_work_stress, df_all_work_work_per_day, title="All work Stress and Active Assignments Over Time")
    
    ## Cache data
    # save_pickle(df_all_work,"data_cached/df_all_work.pkl")
    # df_all_work = load_pickle("data_cached/df_all_work.pkl")
    
    # save_pickle(df_all_work_work_per_day,"data_cached/df_all_work_work_per_day.pkl")
    # df_all_work_work_per_day = load_pickle("data_cached/df_all_work_work_per_day.pkl")
    
    # save_pickle(df_all_work_stress,"data_cached/df_all_work_stress.pkl")
    # df_all_work_stress = load_pickle("data_cached/df_all_work_st/ress.pkl")
    
    '''
    Sleep Data
    '''
    
    ## Initliaze locaction dataframe
    # Load
    with open("sleep_data/loction_timeline_redacted.json", "r") as file:
        data = json.load(file)

    # Convert to df
    df_loc = pd.DataFrame(data, columns=["startTime", "endTime"])

    # Transform Day-Time
    df_loc = df_loc.melt(value_name="Day-Time", var_name="Type")
    df_loc["Day-Time"] = pd.to_datetime(df_loc["Day-Time"])
    df_loc["Day-Time"] = df_loc["Day-Time"].dt.strftime('%m/%d/%Y %I:%M:%S %p')

    df_loc = init_transformation(df_loc, "Day-Time", compare_algo="earliest_pt")

    # # Original Plot
    # plot_start_end_activity(df_loc)

    # Filter extreme start/finish times
    df_loc = filter_extraneous_times(df_loc)
    
    ## Initialize search dataframe
    file_path = 'sleep_data/all_browser_search_redacted.xlsx'
    column_name = 'Day-Time'
    df_search = pd.read_excel(file_path, usecols=[0])
    # df_search = init_transformation(df_search, column_name, compare_algo="diff_compare")
    df_search = init_transformation(df_search, column_name, compare_algo="earliest_pt")

    # plot_start_end_activity(df_search, x_label="month")       # Plot raw sleep data with labels
    
    df_search = filter_extraneous_times(df_search)
    
    # Combine
    df_loc['Date'] = pd.to_datetime(df_loc['Date'])
    df_search['Date'] = pd.to_datetime(df_search['Date'])
    df_all_sleep = update_df_start_points([df_loc, df_search]) 
    
    # Set date range
    df_all_sleep = filter_by_date_custom(df_all_sleep, df_all_work['assigned_date'].min(), df_all_work['due_date'].max())
    # df_all_sleep = filter_by_date_custom(df_all_sleep, "2021-07-01", "2025-07-01")

    # Plot final sleep data
    plot_start_end_activity(df_all_sleep, x_label="month")    
        
    df_sleep_times = get_sleep_hours(df_all_sleep)
    df_sleep_times = df_sleep_times[df_sleep_times["time_slept"] <= 12]     # Filter extraneous sleep times
        
    plot_sleep_hours(df_sleep_times)
    
    # Get Unique Sleep records - Testing
    # dates_count = df_sleep_times.shape[0]  # Count total records
    # unique_dates_count = df_sleep_times["LogicalDate"].nunique()  # Count unique LogicalDate records
    # print("Number of unique LogicalDate records:", unique_dates_count, "Total records:", dates_count)
    
    ## Cache data
    # save_pickle(df_sleep_times,"data_cached/df_sleep_times.pkl")
    # df_sleep_times = load_pickle("data_cached/df_sleep_times.pkl")
    
    '''
    Gaming Data
    '''
    df_gaming = extract_gaming_excel("gaming_data/matches.xlsx")
    df_gaming_time_per_day = get_gaming_time_per_day(df_gaming)
    plot_scatter_per_date(df_gaming_time_per_day, "total_gaming_time_hours", title="Gaming Time per Day", ylabel="Gaming Time (hrs)")
    print(f'N = {len(df_gaming_time_per_day)}')
    
    ## Cache data
    # # save_pickle(df_gaming_time_per_day,"data_cached/df_gaming_time_per_day.pkl")
    # df_gaming_time_per_day = load_pickle("data_cached/df_gaming_time_per_day.pkl")
    
    '''
    Fitness Data
    '''
    df_fitness = extract_fitness_xml("phone_data/export.xml")
    
    # Steps
    df_steps_per_day = fitness_measure_per_day(df_fitness, record_name="HKQuantityTypeIdentifierStepCount", measure_label="step_count")
    plot_scatter_per_date(df_steps_per_day, "step_count", "# Steps per Day", ylabel="Step Count")
    print(f'df_steps_per_day N={len(df_steps_per_day)}')
    
    # Active energy burned
    df_active_burn_per_day = fitness_measure_per_day(df_fitness, record_name="HKQuantityTypeIdentifierActiveEnergyBurned", measure_label="active_burn_Cal")
    plot_scatter_per_date(df_active_burn_per_day, "active_burn_Cal", "Active Burn per Day", ylabel="Active Burn (Cal)")
    print(f'df_active_burn_per_day N={len(df_active_burn_per_day)}')
    
    # Basal energy burned
    df_basal_burn_per_day = fitness_measure_per_day(df_fitness, record_name="HKQuantityTypeIdentifierBasalEnergyBurned", measure_label="basal_burn_Cal")
    plot_scatter_per_date(df_basal_burn_per_day, "basal_burn_Cal", "Basal Burn per Day", ylabel="Basal Burn (Cal)")
    print(f'df_basal_burn_per_day N={len(df_basal_burn_per_day)}')
    
    ## Cache data
    # save_pickle(df_steps_per_day,"data_cached/df_steps_per_day.pkl")
    # df_steps_per_day = load_pickle("data_cached/df_steps_per_day.pkl")
    
    # save_pickle(df_active_burn_per_day,"data_cached/df_active_burn_per_day.pkl")
    # df_active_burn_per_day = load_pickle("data_cached/df_active_burn_per_day.pkl")
    
    # save_pickle(df_basal_burn_per_day,"data_cached/df_basal_burn_per_day.pkl")
    # df_basal_burn_per_day = load_pickle("data_cached/df_basal_burn_per_day.pkl")
    
    '''
    Life Time-series Correlations
    - Pearson Coefficient
    - P value
    '''
    ### Time-series correlations
    
    # Format date
    df_all_work_stress["date"] = pd.to_datetime(df_all_work_stress["date"]).dt.date
    
    ## Stress vs sleep (hr)
    df_sleep_times["LogicalDate"] = pd.to_datetime(df_sleep_times["LogicalDate"]).dt.date  # Extract only the date part
    df_stress_vs_sleep = df_sleep_times.merge(df_all_work_stress, left_on="LogicalDate", right_on="date")
    df_stress_vs_sleep = df_stress_vs_sleep.drop(columns=['LogicalDate'], errors='ignore')
    df_stress_vs_sleep = df_stress_vs_sleep[df_stress_vs_sleep["stress"] != 0].copy()             # Remove stress = 0 case
    corr_coefficient, p_value = pearsonr(df_stress_vs_sleep["stress"], df_stress_vs_sleep["time_slept"])
        
    print(f"sleep pearson corr {corr_coefficient:.2f}, P value: {p_value}, sample size {len(df_stress_vs_sleep)}")
    plot_scatter_compare(df_stress_vs_sleep, "stress", "time_slept", title="Sleep per Day vs Stress Level")
    
    ## Stress vs steps
    df_steps_per_day["date"] = pd.to_datetime(df_steps_per_day["date"]).dt.date  # Extract only the date part
    df_stress_vs_steps = df_steps_per_day.merge(df_all_work_stress, left_on="date", right_on="date")
    df_stress_vs_steps = df_stress_vs_steps[df_stress_vs_steps["stress"] != 0].copy()             # Remove stress = 0 case
    corr_coefficient, p_value = pearsonr(df_stress_vs_steps["stress"], df_stress_vs_steps["step_count"])
        
    print(f"step_count pearson corr {corr_coefficient:.2f}, P value: {p_value}, sample size {len(df_stress_vs_steps)}")
    plot_scatter_compare(df_stress_vs_steps, "stress", "step_count", title="Steps per Day vs Stress Level")
    
    ## Stress vs active burn
    df_active_burn_per_day["date"] = pd.to_datetime(df_active_burn_per_day["date"]).dt.date  # Extract only the date part
    df_stress_vs_active_burn = df_active_burn_per_day.merge(df_all_work_stress, left_on="date", right_on="date")
    df_stress_vs_active_burn = df_stress_vs_active_burn[df_stress_vs_active_burn["stress"] != 0].copy()             # Remove stress = 0 case
    corr_coefficient, p_value = pearsonr(df_stress_vs_active_burn["stress"], df_stress_vs_active_burn["active_burn_Cal"])
    
    print(f"active_burn_Cal pearson corr {corr_coefficient:.2f}, P value: {p_value}, sample size {len(df_stress_vs_active_burn)}")
    plot_scatter_compare(df_stress_vs_active_burn, "stress", "active_burn_Cal", title="Active Burn (Cal) per Day vs Stress Level")
    
    ## Stress vs basal burn
    df_basal_burn_per_day["date"] = pd.to_datetime(df_basal_burn_per_day["date"]).dt.date  # Extract only the date part
    df_stress_vs_basal_burn = df_basal_burn_per_day.merge(df_all_work_stress, left_on="date", right_on="date")
    df_stress_vs_basal_burn = df_stress_vs_basal_burn[df_stress_vs_basal_burn["stress"] != 0].copy()             # Remove stress = 0 case
    corr_coefficient, p_value = pearsonr(df_stress_vs_basal_burn["stress"], df_stress_vs_basal_burn["basal_burn_Cal"])
    
    print(f"basal_burn_Cal pearson corr {corr_coefficient:.2f}, P value: {p_value}, sample size {len(df_stress_vs_basal_burn)}")
    plot_scatter_compare(df_stress_vs_basal_burn, "stress", "basal_burn_Cal", title="Basal Burn (Cal) per Day vs Stress Level")
    
    ## Stress vs Gaming
    df_gaming_time_per_day["date"] = pd.to_datetime(df_gaming_time_per_day["date"]).dt.date  # Extract only the date part
    df_gaming_time_per_day = df_gaming_time_per_day.merge(df_all_work_stress, left_on="date", right_on="date")
    df_stress_vs_gaming = df_gaming_time_per_day[df_gaming_time_per_day["stress"] != 0].copy()             # Remove stress = 0 case
    corr_coefficient, p_value = pearsonr(df_stress_vs_gaming["stress"], df_stress_vs_gaming["total_gaming_time_hours"])
    
    print(f"gaming pearson corr {corr_coefficient:.2f}, P value: {p_value}, sample size {len(df_stress_vs_gaming)}")
    plot_scatter_compare(df_stress_vs_gaming, "stress", "total_gaming_time_hours", title="Gaming Time per Day vs Stress Level")
    
    '''
    Life Time-series Causation Modeling
    - Ordinary Least Squares (OLS) Regression Modeling
        - Model all variables (except target) against target
        - Model just stress against target
    '''    
    
    # Combine all variables - Uncomment when needed
    df_all_variables = (
        df_all_work_stress
        # .merge(df_steps_per_day, on="date")
        # .merge(df_active_burn_per_day, on="date")
        # .merge(df_basal_burn_per_day, on="date")
        # .merge(df_stress_vs_sleep, on="date")     # Limited samples may want to comment
        .merge(df_gaming_time_per_day, on="date")
        # Add more...
    )
    
    # Weird exception when adding sleep df
    if "stress_x" in df_all_variables.columns and "stress_y" in df_all_variables.columns:
        df_all_variables["stress"] = df_all_variables[["stress_x", "stress_y"]].mean(axis=1)
        df_all_variables.drop(columns=["stress_x", "stress_y"], inplace=True)
    
    df_all_variables = df_all_variables[df_all_variables["stress"] != 0].copy()
    
    print(df_all_variables.head(5))
    print(len(df_all_variables))

    # Get all column names
    # all_features = ['active_burn_Cal', 'basal_burn_Cal', 'stress', 'step_count', 'total_gaming_time_hours', 'time_slept']  # Update...
    all_features = ['total_gaming_time_hours', 'stress']  # Update...

    # Loop through each feature as target
    for target in all_features:
        print(f"\n#### Target Variable: {target} ####")

        # Define independent variables (exclude target)
        predictors = [col for col in all_features if col != target]

        # OLS with just "stress"
        X_stress = df_all_variables[['stress']]
        X_stress = sm.add_constant(X_stress)  # Add intercept
        model_stress = sm.OLS(df_all_variables[target], X_stress).fit()
        print("\n### OLS Regression (Stress Only) ###")
        print(model_stress.summary())

        # OLS with all predictors except target
        X_all = df_all_variables[predictors]
        X_all = sm.add_constant(X_all)  # Add intercept
        model_all = sm.OLS(df_all_variables[target], X_all).fit()
        print("\n### OLS Regression (All Variables) ###")
        print(model_all.summary())