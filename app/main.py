import streamlit as st
import pandas as pd
import plotly.express as px
from db_utils import call_filter_employees

st.set_page_config(page_title="Employee Compensation Tool", layout="wide")
st.title("🔍 Employee Filter & Compensation Viewer")

# --- Fetch all employee data for dropdowns ---
all_data = call_filter_employees(None, None, True)
all_roles = sorted(all_data['role'].unique())
all_locations = sorted(all_data['location'].unique())

# --- Sidebar: Filters and Increment Options ---
st.sidebar.header("Filter Options")
selected_roles = st.sidebar.multiselect("Select Role(s)", options=all_roles, key='selected_roles')
selected_locations = st.sidebar.multiselect("Select Location(s)", options=all_locations, key='selected_locations')
include_inactive = st.sidebar.checkbox("Include Inactive Employees", value=True)

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
    filtered_data = call_filter_employees(None, None, include_inactive)
    employee_names = sorted(filtered_data['name'].unique())
    selected_employees = st.sidebar.multiselect("Select Employees", options=employee_names)
    for emp in selected_employees:
        employee_increments[emp] = st.sidebar.slider(f"% Increment for {emp}", 0, 50, 10)

# --- Fetch Filtered Employee Data ---
df = call_filter_employees(None, None, include_inactive)
if selected_roles:
    df = df[df['role'].isin(selected_roles)]
if selected_locations:
    df = df[df['location'].isin(selected_locations)]

# Detect if filters changed
current_filter_hash = hash((tuple(selected_roles), tuple(selected_locations), include_inactive))
if 'filter_hash' not in st.session_state:
    st.session_state['filter_hash'] = current_filter_hash

if current_filter_hash != st.session_state['filter_hash']:
    st.session_state['sim_df'] = pd.DataFrame()
    st.session_state['increment_applied'] = False
    st.session_state['filter_hash'] = current_filter_hash

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
avg_current = sim_df['current_comp_inr'].mean()
avg_updated = sim_df['updated_compensation'].mean() if 'updated_compensation' in sim_df.columns else avg_current
diff = avg_updated - avg_current
st.markdown("### 📌 Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", total_employees)
col2.metric("Avg. Current Compensation", f"₹{avg_current:,.0f}")
col3.metric("Avg. Updated Compensation", f"₹{avg_updated:,.0f}", delta=f"₹{diff:,.0f}")

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

        st.markdown("### 📊 Location-wise Average Pay: Pre vs Post Raise")
        chart_long = chart_data.melt(id_vars='location', var_name='Type', value_name='Average Compensation')
        fig = px.bar(chart_long, x='location', y='Average Compensation', color='Type', barmode='group',
                     text_auto='.2s', height=500)
        fig.update_traces(
            hovertemplate='Location: %{x}<br>Average Compensation: ₹%{y:.2f}<extra></extra>'
        )
        fig.update_layout(xaxis_title='Location', yaxis_title='Average Compensation in ₹',
                          legend_title='Compensation Type', uniformtext_minsize=8, uniformtext_mode='hide', dragmode=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("📊 Experience Band Distribution")
    exp_summary = sim_df['years_of_experience'].value_counts().sort_index()
    exp_df = pd.DataFrame({"experience_band": exp_summary.index, "employee_count": exp_summary.values})
    #st.dataframe(exp_df)
    st.bar_chart(exp_df.rename(columns={"experience_band": "index"}).set_index("index"))

    st.markdown("### 📁 Download Filtered or Simulated Data")
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
