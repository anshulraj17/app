import streamlit as st
import pandas as pd
import plotly.express as px
from db_utils import call_filter_employees, get_turnover_rates, get_industry_benchmarks, get_employee_rating_diff

st.set_page_config(page_title="TechSolve Workforce Insights Dashboard", layout="wide")

# --- UI Styling ---
page_styles = """
<style>
html, body, [class*="css"] {
    font-family: 'Helvetica', sans-serif;
    
}

/* Main Dashboard Title */
h1 {
    font-size: 32px !important;
    font-weight: 800 !important;
}

/* KPI Big Numbers */
[data-testid="stMetricValue"] {
    font-size: 32px !important;
    font-weight: 800 !important;
}

/* KPI Labels/Descriptions */
[data-testid="stMetricLabel"] {
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* Section Headers (e.g., Graph/Table Titles) */
h3, h2 {
    font-size: 22px !important;
    font-weight: 600 !important;
}

/* Table Column Headers */
thead tr th {
    font-size: 20px !important;
    font-weight: bold !important;
}

/* Table Data */
tbody tr td {
    font-size: 20px !important;
    font-weight: normal !important;
}

/* Graph Axis Titles 
g.xtick text, g.ytick text {
    font-size: 14px !important;
    font-weight: 500 !important;
}*/

/* Graph Axis Legends 
.legend text {
    font-size: 13px !important;
}*/

/* General Text/Explanations */
p, span, label, .markdown-text-container {
    font-size: 13px !important;
    font-weight: normal !important;
}

/* Apply consistent background */
.block-container {
    background-color: #FAFAFA;
    padding: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #C8DAFF;
}
</style>
"""
st.markdown(page_styles, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #C8DAFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    "<h1 style='text-align: center;'>Workforce Insights Dashboard</h1>",
    unsafe_allow_html=True
)
# --- Fetch all employee data for dropdowns ---
all_data = call_filter_employees(None, None, True)
all_roles = sorted(all_data['role'].unique())
all_locations = sorted(all_data['location'].unique())

# --- Sidebar: Filters and Increment Options ---
st.sidebar.image("assets/logo.png", use_container_width=True)
st.sidebar.header("Filter Options")
employee_status_option = st.sidebar.radio(
    "Select Employee Status",
    options=["All Employees", "Active Employees", "Inactive Employees"],
    index=0
)
selected_roles = st.sidebar.multiselect("Select Role(s)", options=all_roles, key='selected_roles')
selected_locations = st.sidebar.multiselect("Select Location(s)", options=all_locations, key='selected_locations')
# --- Experience Band Filter ---
experience_bands = sorted(all_data['years_of_experience'].dropna().unique())
selected_exp_band = st.sidebar.selectbox("Select Experience Band", options=["All"] + experience_bands, index=0)

st.sidebar.markdown("---")
st.sidebar.header("Increment Options")
increment_type = st.sidebar.radio("Choose Increment Type:", ["Global", "Custom by Location", "Custom by Employee"])

# --- Global Increment Input ---
global_increment = st.sidebar.slider("Global Increment %", min_value=0, max_value=50, value=0)

# --- Custom by Location Inputs ---
custom_increments = {}
if increment_type == "Custom by Location":
    selected_locs = st.sidebar.multiselect("Select Locations", options=all_locations)
    for loc in selected_locs:
        custom_increments[loc] = st.sidebar.slider(f"% Increment for {loc}", 0, 50, 10)

# --- Custom by Employee Inputs ---
employee_increments = {}
if increment_type == "Custom by Employee":
    if employee_status_option == "All Employees":
        filtered_data = call_filter_employees(None, None, True)
    elif employee_status_option == "Active Employees":
        filtered_data = call_filter_employees(None, None, False)
        filtered_data = filtered_data[filtered_data['active'] == 1]
    elif employee_status_option == "Inactive Employees":
        filtered_data = call_filter_employees(None, None, True)
        filtered_data = filtered_data[filtered_data['active'] == 0]

    employee_names = sorted(filtered_data['name'].unique())
    selected_employees = st.sidebar.multiselect("Select Employees", options=employee_names)
    for emp in selected_employees:
        employee_increments[emp] = st.sidebar.slider(f"% Increment for {emp}", 0, 50, 10)

# Fetch data based on employee status
if employee_status_option == "All Employees":
    df = call_filter_employees(None, None, True)
elif employee_status_option == "Active Employees":
    df = call_filter_employees(None, None, False)
    df = df[df['active'] == 1]
elif employee_status_option == "Inactive Employees":
    df = call_filter_employees(None, None, True)
    df = df[df['active'] == 0]

# Merge with rating difference data (consistently for all status options)
rating_diff_df = get_employee_rating_diff()
df = pd.merge(df, rating_diff_df[['name', 'self_rating_l3q', 'manager_rating_l3q', 'self_greater_than_manager']],
              how='left', on='name')
    
if selected_roles:
    df = df[df['role'].isin(selected_roles)]
if selected_locations:
    df = df[df['location'].isin(selected_locations)]

# Detect if filters changed
current_filter_hash = hash((tuple(selected_roles), tuple(selected_locations), employee_status_option))
if 'filter_hash' not in st.session_state:
    st.session_state['filter_hash'] = current_filter_hash

if current_filter_hash != st.session_state['filter_hash']:
    st.session_state['sim_df'] = pd.DataFrame()
    st.session_state['increment_applied'] = False
    st.session_state['filter_hash'] = current_filter_hash

if selected_exp_band != "All":
    df = df[df['years_of_experience'] == selected_exp_band]

# Apply Increment Button
if st.sidebar.button("Apply Increment") and not df.empty:
    sim_df = df.copy()
    sim_df['current_compensation'] = sim_df['current_comp_inr'].astype(float)

    if increment_type == "Global":
        sim_df['updated_compensation'] = (sim_df['current_compensation'] * (1 + global_increment / 100)).round(2)
    elif increment_type == "Custom by Location":
        sim_df['updated_compensation'] = sim_df.apply(
            lambda row: (row['current_compensation'] * (1 + custom_increments.get(row['location'], 0) / 100))
            if row['location'] in custom_increments else row['current_compensation'], axis=1
        ).round(2)
    elif increment_type == "Custom by Employee":
        sim_df['updated_compensation'] = sim_df.apply(
            lambda row: (row['current_compensation'] * (1 + employee_increments.get(row['name'], 0) / 100))
            if row['name'] in employee_increments else row['current_compensation'], axis=1
        ).round(2)

    st.session_state['sim_df'] = sim_df
    st.session_state['increment_applied'] = True

# Select active DataFrame
sim_df = st.session_state['sim_df'] if st.session_state.get('increment_applied') else df

# --- Summary Cards ---

total_employees = len(sim_df)
total_current = float(sim_df['current_comp_inr'].sum())
total_updated = float(sim_df['updated_compensation'].sum()) if 'updated_compensation' in sim_df.columns else total_current
avg_current = sim_df['current_comp_inr'].mean()
avg_updated = sim_df['updated_compensation'].mean() if 'updated_compensation' in sim_df.columns else avg_current
diff_avg = avg_updated - avg_current
diff_total = total_updated - total_current

st.markdown(
    "<br><h3 style='text-align: center;'>Key Metrics</h3>",
    unsafe_allow_html=True
)
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
col1.metric("Total Employees", total_employees)
col2.metric("Total Current Compensation", f"₹{total_current:,.0f}")
col3.metric("Total Updated Compensation", f"₹{total_updated:,.0f}", delta=f"₹{diff_total:,.0f}")
col4.metric("Avg. Current Compensation", f"₹{avg_current:,.0f}")
col5.metric("Avg. Updated Compensation", f"₹{avg_updated:,.0f}", delta=f"₹{diff_avg:,.0f}")

# --- Display Filtered/Simulated Data ---
if sim_df.empty:
    st.warning("No data found for the selected filters.")
else:
    st.success(f"Found {len(sim_df)} employee(s).")
    st.dataframe(sim_df)

    if 'location' in sim_df.columns:
        chart_data = sim_df.groupby("location").agg(
            current_avg=('current_comp_inr', 'mean'),
            updated_avg=('updated_compensation', 'mean') if 'updated_compensation' in sim_df.columns else ('current_comp_inr', 'mean')
        ).reset_index()

        st.markdown("### 💵 Location-wise Average Pay: Pre vs Post Raise")
        chart_long = chart_data.melt(id_vars='location', var_name='Type', value_name='Average Compensation')
        fig = px.bar(chart_long, x='location', y='Average Compensation', color='Type', barmode='group', text_auto='.2s', height=500, color_discrete_map={'current_avg': '#FFE6E1', 'updated_avg': '#075B5E'})

        fig.update_traces(
            hovertemplate='Location: %{x}<br>Average Compensation: ₹%{y:.2f}<extra></extra>'
        )
        fig.update_layout(
            title={
                'text': "Location-wise Average Pay: Pre vs Post Raise",
                'x': 0.4,
                'font': dict(size=20)
            },
            xaxis=dict(
                title='Location',
                title_font=dict(size=20),
                tickfont=dict(size=20)
            ),
            yaxis=dict(
                title="Average Compensation (₹ '000)",
                title_font=dict(size=20),
                tickfont=dict(size=20)
            ),
            legend=dict(
                title='Compensation Type',
                font=dict(size=20),
                title_font=dict(size=20)
            ),
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            dragmode=False
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
st.markdown("### 📈 Experience Band Distribution")

if employee_status_option == "All Employees":
    # Group filtered sim_df correctly
    exp_grouped = sim_df.groupby(['years_of_experience', 'active']).agg(
        employee_count=('name', 'count'),
        avg_compensation=('current_comp_inr', 'mean')
    ).reset_index()

    exp_grouped['Status'] = exp_grouped['active'].map({1: "Active", 0: "Inactive"})

    fig_exp = px.bar(
        exp_grouped,
        x='years_of_experience',
        y='employee_count',
        color='Status',
        barmode='group',
        title="Experience Band Distribution by Employee(s) Status",
        custom_data=['avg_compensation', 'Status'],
        color_discrete_map={
            "Active": "#8ACCD5",     
            "Inactive": "#E55050"    
    }

    )


    # Center-align the title
    fig_exp.update_layout(title_x=0.4)

    fig_exp.update_traces(
        hovertemplate=(
            'Experience Band: %{x}<br>' +
            'Status: %{customdata[1]}<br>' +
            'Employee(s) Count: %{y}<br>' +
            'Avg. Compensation: ₹%{customdata[0]:,.2f}<extra></extra>'
        )
    )

    fig_exp.update_layout(
        title={
            'text': "Experience Band Distribution by Employee(s) Status",
            'x': 0.4,
            'font': dict(size=20)
        },
        xaxis_title='Experience Band',
        yaxis_title='Employee(s) Count',
        height=400,
        xaxis=dict(
            title_font=dict(size=20),
            tickfont=dict(size=20)
        ),
        yaxis=dict(
            title_font=dict(size=20),
            tickfont=dict(size=20)
        ),
        legend=dict(
            font=dict(size=20)
        )
    )




    st.plotly_chart(fig_exp, use_container_width=True)

else:
    exp_grouped = sim_df.groupby('years_of_experience').agg(
        employee_count=('name', 'count'),
        avg_compensation=('current_comp_inr', 'mean')
    ).reset_index()

    fig_exp_single = px.bar(
        exp_grouped,
        x='years_of_experience',
        y='employee_count',
        title="Experience Band Distribution",
        custom_data=['avg_compensation']
    )

    fig_exp_single.update_traces(
        hovertemplate=(
            'Experience Band: %{x}<br>' +
            'Employee(s) Count: %{y}<br>' +
            'Avg. Compensation: ₹%{customdata[0]:,.2f}<extra></extra>'
        )
    )

    fig_exp_single.update_layout(
        xaxis_title='Experience Band',
        yaxis_title='Employee(s) Count',
        height=400
    )
    st.plotly_chart(fig_exp_single, use_container_width=True)

# --- Turnover Analysis Section ---
st.markdown("---")
st.markdown("### 🔄 Employees Turnover Analysis")

turnover_df = get_turnover_rates()

if turnover_df.empty:
    st.warning("No turnover data found.")
else:

    # Location filter
    turnover_locations = sorted(turnover_df['location_name'].unique())
    selected_location = st.selectbox("Choose Location", options=["All"] + turnover_locations, index=0)

    # Role filter
    turnover_roles = sorted(turnover_df['role_name'].unique())
    selected_role = st.selectbox("Choose Role", options=["All"] + turnover_roles, index=0)

    # Apply filters
    filtered_turnover_df = turnover_df.copy()
    if selected_location != "All":
        filtered_turnover_df = filtered_turnover_df[filtered_turnover_df['location_name'] == selected_location]
    if selected_role != "All":
        filtered_turnover_df = filtered_turnover_df[filtered_turnover_df['role_name'] == selected_role]

    # Display Table
    st.dataframe(filtered_turnover_df)

    # Bar chart
    fig_turnover = px.bar(
        filtered_turnover_df.sort_values("turnover_percentage", ascending=False),
        x="turnover_percentage",
        y="role_name",
        color="location_name",
        orientation="h",
        title="Turnover Rate by Role & Location", 
        color_discrete_map={'Bangalore':'#3E3F5B', 'Jaipur': '#FF90BB','Pune':'#D29F80'},
        text="turnover_percentage"
    )
    fig_turnover.update_layout(
        title={
            'text': "Turnover Rate by Role & Location",
            'x': 0.4,
            'font': dict(size=20)
        },
        xaxis=dict(
            title="Turnover Percentage",
            title_font=dict(size=20),
            tickfont=dict(size=20)
        ),
        yaxis=dict(
            title="Role",
            title_font=dict(size=20),
            tickfont=dict(size=20)
        ),
        legend=dict(
            title="Location",
            font=dict(size=20),
            title_font=dict(size=20)
        ),
        height=500
    )

    fig_turnover.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside',
        insidetextfont=dict(size=20),
        outsidetextfont=dict(size=20)
    )
    st.plotly_chart(fig_turnover, use_container_width=True)

# Center-align the title
from db_utils import get_industry_benchmarks
st.markdown("---")
st.markdown("### 🏢 Industry Benchmark Comparison")

# Get benchmark data
industry_df = get_industry_benchmarks()

industry_df = industry_df.rename(columns={
    'role_name': 'role',
    'location_name': 'location',
    'average_industry_comp': 'industry_avg_compensation'
})

# Apply sidebar filters
filtered_emp_df = sim_df.copy()
filtered_roles = selected_roles if selected_roles else all_roles
filtered_locations = selected_locations if selected_locations else all_locations

# Employee summary
emp_comp_summary = filtered_emp_df.groupby(['role', 'location']).agg(
    employee_avg_comp=('current_comp_inr', 'mean')
).reset_index()

# Filter industry benchmark
filtered_industry_df = industry_df[
    (industry_df['role'].isin(filtered_roles)) &
    (industry_df['location'].isin(filtered_locations))
]

# Merge for comparison
benchmark_compare_df = pd.merge(
    emp_comp_summary,
    filtered_industry_df,
    on=['role', 'location'],
    how='inner'
)

if benchmark_compare_df.empty:
    st.warning("No benchmark comparison data available for selected filters.")
else:
    compare_long = benchmark_compare_df.melt(
        id_vars=['role', 'location'],
        value_vars=['employee_avg_comp', 'industry_avg_compensation'],
        var_name='Source',
        value_name='Average Compensation'
    )

    compare_long['Source'] = compare_long['Source'].map({
        'employee_avg_comp': 'TechSolveEmployee',
        'industry_avg_compensation': 'Industry Benchmark'
    })

    # Convert to millions and round to 2 decimal places
    compare_long["Comp_M"] = compare_long["Average Compensation"].astype(float) / 1_000_000
    compare_long["Comp_M"] = compare_long["Comp_M"].round(2)

    # Custom text formatting for display
    compare_long["Display_Label"] = compare_long["Comp_M"].apply(lambda x: f"{x:.2f}M")

    fig_benchmark = px.bar(
        compare_long,
        x='role',
        y='Comp_M',
        color='Source',
        barmode='group',
        facet_col='location',
        text='Display_Label',
        title="Avg. Compensation: TechSolve vs Industry Benchmark",
        color_discrete_map={
            'TechSolveEmployee': '#0d6efd',
            'Industry Benchmark': '#adb5bd'
        },
        facet_col_spacing=0.1
    )

    # Correct hovertemplate using actual DataFrame fields
    fig_benchmark.update_traces(
        hovertemplate=(
            'Source: %{customdata[0]}<br>' +
            'Location: %{customdata[1]}<br>' +
            'Role: %{x}<br>' +
            'Average Compensation: ₹%{y:.2f}M<extra></extra>'
        ),
        customdata=compare_long[['Source', 'location']]
    )

    fig_benchmark.update_layout(
        height=500,
        xaxis_title='Role',
        yaxis_title='Average Compensation (₹ Mn)',
        legend_title='Source',
        title_x=0.4
    )

    st.plotly_chart(fig_benchmark, use_container_width=True)

st.markdown("---")
st.markdown("## 💼 Revised Compensation: Bonus vs Stock Simulation")

# Add 'All' to role filter
valid_roles = ["Senior Associate", "Manager"]
roles_sim = ["All"] + valid_roles

locations_sim = ["All"] + sorted(sim_df['location'].dropna().unique().tolist())

selected_sim_role = st.selectbox("Select Role", roles_sim, key="sim_role_select")
selected_sim_location = st.selectbox("Select Location", locations_sim, key="sim_loc_select")

benefit_type = st.radio("Choose Benefit Type", ["Bonus", "Stock Option"], horizontal=True)
input_mode = st.radio("Benefit Application Mode", ["Fixed Amount", "% of Salary"], horizontal=True)

# Define benefit inputs per role
benefit_inputs = {}
if selected_sim_role == "All":
    st.markdown("### 🎯 Enter Benefit for Each Role")
    if input_mode == "% of Salary":
        benefit_inputs["Manager"] = st.slider("Manager – % of Salary", 0, 100, 10, key="mgr_pct")
        benefit_inputs["Senior Associate"] = st.slider("Senior Associate – % of Salary", 0, 100, 10, key="sa_pct")
    else:
        benefit_inputs["Manager"] = st.number_input("Manager – Fixed Amount (₹)", step=10000, value=100000, key="mgr_amt")
        benefit_inputs["Senior Associate"] = st.number_input("Senior Associate – Fixed Amount (₹)", step=10000, value=100000, key="sa_amt")
else:
    if input_mode == "% of Salary":
        benefit_inputs[selected_sim_role] = st.slider(f"{selected_sim_role} – % of Salary", 0, 100, 10)
    else:
        benefit_inputs[selected_sim_role] = st.number_input(f"{selected_sim_role} – Fixed Amount (₹)", step=10000, value=100000)

vesting_period = None
if benefit_type == "Stock Option":
    vesting_period = st.selectbox("Stock Vesting Period (Years)", [1, 2, 3, 4])

# Filter DataFrame
relevant_df = sim_df.copy()
relevant_df = relevant_df[relevant_df['role'].isin(valid_roles)]

if selected_sim_role != "All":
    relevant_df = relevant_df[relevant_df['role'] == selected_sim_role]

if selected_sim_location != "All":
    relevant_df = relevant_df[relevant_df['location'] == selected_sim_location]

# Ensure numeric ops are safe
relevant_df['current_comp_inr'] = relevant_df['current_comp_inr'].astype(float)

if not relevant_df.empty:
    # Initialize revised compensation with base
    relevant_df['revised_compensation'] = relevant_df['current_comp_inr']

    for role in relevant_df['role'].unique():
        mask = relevant_df['role'] == role
        base_salary = relevant_df.loc[mask, 'current_comp_inr']
        value = benefit_inputs.get(role, 0)

        if input_mode == "% of Salary":
            if benefit_type == "Bonus":
                relevant_df.loc[mask, 'revised_compensation'] = base_salary * (1 + value / 100)
            else:  # Stock Option
                relevant_df.loc[mask, 'revised_compensation'] = base_salary + (base_salary * value / 100)
        else:
            if benefit_type == "Bonus":
                relevant_df.loc[mask, 'revised_compensation'] = base_salary + value
            else:  # Stock Option
                relevant_df.loc[mask, 'revised_compensation'] = base_salary + value

    # --- Generate Graph ---
    st.subheader(f"💵 Average Compensation for {selected_sim_role} ({selected_sim_location})")

    avg_current = relevant_df['current_comp_inr'].mean()
    avg_revised = relevant_df['revised_compensation'].mean()

    comp_df = pd.DataFrame({
        "Type": ["Current Average", "Revised Average"],
        "Compensation": [avg_current, avg_revised]
    })

    fig_sim = px.bar(comp_df, x="Type", y="Compensation", text_auto='.2s',
                     title="Current vs Revised Average Compensation" )
    fig_sim.update_layout(
        yaxis_title="Compensation in (₹ Mn)",
        xaxis_title="Type",
        title={
            'text': "Current vs Revised Average Compensation",
            'x': 0.4,
            'font': dict(size=20)
        },
        xaxis=dict(title_font=dict(size=20), tickfont=dict(size=20)),
        yaxis=dict(title_font=dict(size=20), tickfont=dict(size=20)),
        legend=dict(title_font=dict(size=20), font=dict(size=20))  # Optional: for consistency
    )

    fig_sim.update_traces(
        marker_color=["#205781", "#198754"],
        width=0.40

    )



    st.plotly_chart(fig_sim, use_container_width=True)

# --- 📊 Summary Table ---
num_employees = len(relevant_df)
total_current = relevant_df['current_comp_inr'].sum()
total_updated = relevant_df['revised_compensation'].sum()
additional_cost = total_updated - total_current
percent_increase = (additional_cost / total_current) * 100 if total_current != 0 else 0

if benefit_type == "Stock Option":
    annual_pl_impact = additional_cost / vesting_period
else:
    annual_pl_impact = additional_cost

# Constants
annual_revenue = 60e7  # ₹60 Cr in INR

# Calculations for new rows
current_comp_pct_of_revenue = (total_current / annual_revenue) * 100
updated_comp_pct_of_revenue = ((total_current + annual_pl_impact) / annual_revenue) * 100

def format_currency(value, scale='L'):
    if scale == 'Cr':
        return f"₹{value/1e7:.2f} Cr"
    elif scale == 'L':
        return f"₹{value/1e5:.2f} L"
    else:
        return f"₹{value:,.0f}"

summary_data = {
    "Metric": [
        "No. of Employees",
        "Total Current Compensation",
        "Additional Cost",
        "Total Updated Compensation",
        "% Increase in Cost",
        "Annual P & L Impact",
        "Current Compensation Cost (% of Revenue)",
        "Updated Compensation Cost (% of Revenue)"
    ],
    benefit_type + " Scenario": [
        num_employees,
        format_currency(total_current, 'Cr'),
        format_currency(additional_cost, 'L'),
        format_currency(total_updated, 'Cr'),
        f"{percent_increase:.1f}%",
        format_currency(annual_pl_impact, 'L'),
        f"{current_comp_pct_of_revenue:.2f}%",
        f"{updated_comp_pct_of_revenue:.2f}%"
    ]
}

summary_df = pd.DataFrame(summary_data)
st.subheader("📋 Financial Impact Summary")
st.dataframe(summary_df, use_container_width=True)


st.markdown("### 📁 Download Filtered Data")
export_df = sim_df.copy()
export_df = export_df.rename(columns={
        "name": "Name",
        "role": "Role",
        "location": "Location",
        "years_of_experience": "Experience",
        "current_comp_inr": "Current Compensation"
    })

if 'updated_compensation' in export_df.columns:
        export_df = export_df.rename(columns={"updated_compensation": "Updated Compensation"})
else:
        export_df['Updated Compensation'] = ""

export_df['Experience'] = export_df['Experience'].apply(lambda x: f"'{x}" if '-' in str(x) else x)
export_df['Status'] = export_df['active'].map({1: "Active", 0: "Inactive"})
export_df = export_df[["Name", "Role", "Location", "Experience", "Current Compensation", "Updated Compensation", "Status"]]

csv = export_df.to_csv(index=False).encode('utf-8')
st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name='employee_compensation.csv',
        mime='text/csv'
    )



