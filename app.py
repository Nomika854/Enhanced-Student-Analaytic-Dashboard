import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Set page configuration
st.set_page_config(
    page_title="Enhanced Student Analytics Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        margin-top: 10px;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    .student-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load and cache the data
@st.cache_data
def load_data():
    df = pd.read_csv('student_info.csv')
    # Calculate average score for each student
    df['avg_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)
    return df

# Function to generate PDF report
def generate_report(student_data):
    buffer = BytesIO()
    plt.figure(figsize=(10, 6))
    subjects = ['math_score', 'reading_score', 'writing_score']
    scores = [student_data[subject] for subject in subjects]
    plt.bar(subjects, scores)
    plt.title(f"Performance Report - {student_data['name']}")
    plt.savefig(buffer, format='png')
    plt.close()
    return buffer

# Function to calculate performance status
def get_performance_status(scores):
    avg_score = np.mean(scores)
    if avg_score >= 80:
        return "Excellent", "🌟"
    elif avg_score >= 70:
        return "Good", "👍"
    elif avg_score >= 60:
        return "Average", "📊"
    else:
        return "Needs Improvement", "⚠️"

# Load the data
df = load_data()

# Sidebar navigation with role selection
st.sidebar.title("📊 Navigation")
role = st.sidebar.selectbox("Select Role", ["Admin", "Faculty", "Parent"])
st.sidebar.markdown("---")

# Custom filter panel
with st.sidebar.expander("📌 Filters"):
    grade_filter = st.multiselect("Grade Level", sorted(df['grade_level'].unique()))
    gender_filter = st.multiselect("Gender", df['gender'].unique())
    score_range = st.slider("Score Range", 0, 100, (0, 100))

# Apply filters
filtered_df = df.copy()
if grade_filter:
    filtered_df = filtered_df[filtered_df['grade_level'].isin(grade_filter)]
if gender_filter:
    filtered_df = filtered_df[filtered_df['gender'].isin(gender_filter)]
filtered_df = filtered_df[
    (filtered_df['math_score'].between(score_range[0], score_range[1])) &
    (filtered_df['reading_score'].between(score_range[0], score_range[1])) &
    (filtered_df['writing_score'].between(score_range[0], score_range[1]))
]

# Main navigation
page = st.sidebar.radio("Choose a page", 
    ["Dashboard", "Subject Analysis", "Attendance Insights", 
     "Student Profiles", "Reports & Downloads", "Behavioral Analytics"])

if page == "Dashboard":
    st.title("📚 Enhanced Student Analytics Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(filtered_df))
    with col2:
        pass_rate = (filtered_df['final_result'] == 'Pass').mean() * 100
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
    with col3:
        avg_attendance = filtered_df['attendance_rate'].mean()
        st.metric("Avg Attendance", f"{avg_attendance:.1f}%")
    with col4:
        at_risk_count = len(filtered_df[
            (filtered_df['attendance_rate'] < 85) & 
            (filtered_df['avg_score'] < 60)
        ])
        st.metric("At Risk Students", at_risk_count)
    
    # Top Performers and At-Risk Students
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌟 Top Performers")
        # Get top 5 students based on average score
        top_performers = filtered_df.nlargest(5, 'avg_score')
        st.dataframe(top_performers[['name', 'avg_score', 'math_score', 'reading_score', 'writing_score']])
        
    with col2:
        st.subheader("⚠️ Students Needing Support")
        at_risk = filtered_df[
            (filtered_df['attendance_rate'] < 85) & 
            (filtered_df['avg_score'] < 60)
        ]
        st.dataframe(at_risk[['name', 'attendance_rate', 'avg_score', 'math_score', 'reading_score', 'writing_score']])

elif page == "Subject Analysis":
    st.title("📊 Subject-Wise Performance Analysis")
    
    # Subject Performance Distribution
    st.subheader("Subject Score Distributions")
    fig = go.Figure()
    for subject in ['math_score', 'reading_score', 'writing_score']:
        fig.add_trace(go.Violin(x=[subject.split('_')[0]]*len(filtered_df),
                               y=filtered_df[subject],
                               name=subject.split('_')[0].capitalize()))
    fig.update_layout(title="Score Distribution Across Subjects",
                     xaxis_title="Subject",
                     yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)
    
    # Subject Correlation Matrix
    st.subheader("Subject Correlation Analysis")
    corr_matrix = filtered_df[['math_score', 'reading_score', 'writing_score']].corr()
    fig_corr = px.imshow(corr_matrix,
                        labels=dict(color="Correlation"),
                        title="Subject Score Correlations")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Subject-wise Performance by Gender
    st.subheader("Gender-wise Subject Performance")
    subject_gender = pd.melt(filtered_df, 
                           value_vars=['math_score', 'reading_score', 'writing_score'],
                           id_vars=['gender'])
    fig_gender = px.box(subject_gender, x='variable', y='value', color='gender',
                       title="Subject Performance by Gender")
    st.plotly_chart(fig_gender, use_container_width=True)

elif page == "Attendance Insights":
    st.title("📅 Attendance vs Performance Analysis")
    
    # Attendance vs Average Score
    st.subheader("Attendance Impact on Performance")
    fig_attendance = px.scatter(filtered_df, x='attendance_rate', y='avg_score',
                              color='final_result', title="Attendance vs Average Score",
                              trendline="ols")
    st.plotly_chart(fig_attendance, use_container_width=True)
    
    # Attendance Categories
    st.subheader("Performance by Attendance Categories")
    filtered_df['attendance_category'] = pd.qcut(filtered_df['attendance_rate'], 
                                               q=4, 
                                               labels=['Low', 'Medium', 'High', 'Excellent'])
    att_perf = filtered_df.groupby('attendance_category')['avg_score'].mean()
    fig_att_cat = px.bar(att_perf, title="Average Score by Attendance Category")
    st.plotly_chart(fig_att_cat, use_container_width=True)

elif page == "Student Profiles":
    st.title("👨‍🎓 Interactive Student Profiles")
    
    # Student Selection
    selected_student = st.selectbox("Select Student", filtered_df['name'].unique())
    student_data = filtered_df[filtered_df['name'] == selected_student].iloc[0]
    
    # Student Profile Card
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📋 Basic Information")
        st.markdown(f"""
        * **Name:** {student_data['name']}
        * **Gender:** {student_data['gender']}
        * **Grade Level:** {student_data['grade_level']}
        * **Parent Education:** {student_data['parent_education']}
        """)
    
    with col2:
        st.markdown("### 📊 Academic Performance")
        scores = [student_data['math_score'], 
                 student_data['reading_score'], 
                 student_data['writing_score']]
        status, emoji = get_performance_status(scores)
        st.markdown(f"""
        * **Math Score:** {student_data['math_score']}
        * **Reading Score:** {student_data['reading_score']}
        * **Writing Score:** {student_data['writing_score']}
        * **Average Score:** {student_data['avg_score']:.2f}
        * **Status:** {status} {emoji}
        """)
    
    with col3:
        st.markdown("### 📈 Other Metrics")
        st.markdown(f"""
        * **Attendance Rate:** {student_data['attendance_rate']:.1f}%
        * **Study Hours:** {student_data['study_hours']:.1f}
        * **Extra Activities:** {student_data['extra_activities']}
        * **Final Result:** {student_data['final_result']}
        """)
    
    # Performance Visualization
    st.subheader("📊 Performance Visualization")
    fig = go.Figure(data=[
        go.Bar(name='Student Scores', 
               x=['Math', 'Reading', 'Writing'],
               y=[student_data['math_score'], 
                  student_data['reading_score'], 
                  student_data['writing_score']]),
        go.Bar(name='Class Average',
               x=['Math', 'Reading', 'Writing'],
               y=[filtered_df['math_score'].mean(),
                  filtered_df['reading_score'].mean(),
                  filtered_df['writing_score'].mean()])
    ])
    fig.update_layout(barmode='group')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Reports & Downloads":
    st.title("📑 Reports and Downloads")
    
    # Report Type Selection
    report_type = st.selectbox("Select Report Type", 
                              ["Individual Student Report", 
                               "Class Performance Report", 
                               "Attendance Report"])
    
    if report_type == "Individual Student Report":
        selected_student = st.selectbox("Select Student", filtered_df['name'].unique())
        student_data = filtered_df[filtered_df['name'] == selected_student].iloc[0]
        
        if st.button("Generate Report"):
            report_buffer = generate_report(student_data)
            st.download_button(
                label="Download Report",
                data=report_buffer,
                file_name=f"report_{selected_student}.pdf",
                mime="application/pdf"
            )
    
    elif report_type == "Class Performance Report":
        if st.button("Download Class Report"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="class_performance.csv",
                mime="text/csv"
            )
    
    elif report_type == "Attendance Report":
        attendance_df = filtered_df[['name', 'attendance_rate', 'grade_level']]
        if st.button("Download Attendance Report"):
            csv = attendance_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="attendance_report.csv",
                mime="text/csv"
            )

else:  # Behavioral Analytics
    st.title("🎯 Behavioral Analytics")
    
    # Study Pattern Analysis
    st.subheader("📚 Study Patterns")
    fig_study = px.scatter(filtered_df, x='study_hours', y='avg_score',
                          color='final_result',
                          title="Study Hours vs Performance")
    st.plotly_chart(fig_study, use_container_width=True)
    
    # Extra Activities Impact
    st.subheader("🎨 Extra Activities Impact")
    extra_act_impact = filtered_df.groupby('extra_activities')['avg_score'].mean()
    fig_extra = px.bar(extra_act_impact,
                      title="Average Score by Extra Activities Participation")
    st.plotly_chart(fig_extra, use_container_width=True)
    
    # Internet Access Analysis
    st.subheader("💻 Internet Access Impact")
    internet_impact = filtered_df.groupby('internet_access')['avg_score'].mean()
    fig_internet = px.bar(internet_impact,
                         title="Average Score by Internet Access")
    st.plotly_chart(fig_internet, use_container_width=True)

# Alerts and Notifications (shown for all pages)
st.sidebar.markdown("---")
st.sidebar.subheader("📢 Alerts")

# Generate alerts based on data
low_attendance = filtered_df[filtered_df['attendance_rate'] < 85]['name'].tolist()
low_performance = filtered_df[filtered_df['avg_score'] < 60]['name'].tolist()

if low_attendance:
    st.sidebar.warning(f"⚠️ {len(low_attendance)} students have attendance below 85%")
if low_performance:
    st.sidebar.warning(f"⚠️ {len(low_performance)} students have low academic performance")

# Role-based access control
if role == "Parent":
    st.sidebar.info("🔒 You are viewing in Parent mode. Only your child's data is visible.")
elif role == "Faculty":
    st.sidebar.info("👨‍🏫 You are viewing in Faculty mode. Class-level data is visible.")
elif role == "Admin":
    st.sidebar.info("👑 You are viewing in Admin mode. Full access granted.")

# Attendance Analysis Section
st.header("Attendance Impact Analysis")

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Attendance vs Performance", "Attendance Patterns", "Statistical Analysis"])

with tab1:
    st.subheader("Attendance vs Academic Performance")
    
    # Scatter plot with trend line
    fig_attendance = px.scatter(filtered_df, 
                              x='attendance_rate', 
                              y='avg_score',
                              trendline="ols",
                              labels={'attendance_rate': 'Attendance Rate (%)', 
                                     'avg_score': 'Average Score'},
                              title='Correlation between Attendance and Performance')
    
    # Add custom hover data
    fig_attendance.update_traces(
        hovertemplate="<br>".join([
            "Attendance: %{x:.1f}%",
            "Score: %{y:.1f}",
        ])
    )
    
    st.plotly_chart(fig_attendance, use_container_width=True)
    
    # Calculate correlation coefficient
    correlation = filtered_df['attendance_rate'].corr(filtered_df['avg_score'])
    st.info(f"Correlation Coefficient: {correlation:.2f}")

with tab2:
    st.subheader("Attendance Distribution")
    
    # Create attendance brackets
    filtered_df['attendance_bracket'] = pd.cut(filtered_df['attendance_rate'], 
                                             bins=[0, 60, 75, 85, 100],
                                             labels=['Critical (<60%)', 'Low (60-75%)', 
                                                    'Good (75-85%)', 'Excellent (>85%)'])
    
    # Create attendance distribution chart
    fig_dist = px.histogram(filtered_df, 
                          x='attendance_bracket',
                          color='attendance_bracket',
                          title='Distribution of Student Attendance',
                          labels={'attendance_bracket': 'Attendance Category', 
                                 'count': 'Number of Students'})
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Show statistics for each bracket
    col1, col2, col3, col4 = st.columns(4)
    brackets = filtered_df['attendance_bracket'].value_counts()
    
    for i, (bracket, count) in enumerate(brackets.items()):
        with [col1, col2, col3, col4][i]:
            st.metric(f"{bracket}", count)

with tab3:
    st.subheader("Statistical Insights")
    
    # Prepare data for regression analysis
    X = filtered_df['attendance_rate'].values.reshape(-1, 1)
    y = filtered_df['avg_score'].values
    
    # Add constant to predictor variables
    X = sm.add_constant(X)
    
    # Fit regression model
    model = sm.OLS(y, X).fit()
    
    # Display regression statistics
    st.write("Regression Analysis Results:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R-squared", f"{model.rsquared:.3f}")
        st.metric("Adjusted R-squared", f"{model.rsquared_adj:.3f}")
    with col2:
        st.metric("F-statistic", f"{model.fvalue:.2f}")
        st.metric("P-value", f"{model.f_pvalue:.4f}")

    # Add recommendations based on analysis
    st.subheader("Recommendations")
    attendance_threshold = 75
    low_attendance = filtered_df[filtered_df['attendance_rate'] < attendance_threshold]
    
    if len(low_attendance) > 0:
        st.warning(f"""
        🚨 {len(low_attendance)} students have attendance below {attendance_threshold}%.
        
        Recommended actions:
        - Schedule parent-teacher meetings for these students
        - Implement attendance improvement strategies
        - Monitor progress weekly
        """)
    
    # Download section
    st.subheader("Export Analysis")
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(filtered_df[['student_id', 'name', 'attendance_rate', 'avg_score', 'attendance_bracket']])
    st.download_button(
        label="Download Attendance Analysis Report",
        data=csv,
        file_name='attendance_analysis.csv',
        mime='text/csv',
    )

# Footer
st.markdown("---")
st.markdown("*Dashboard created for Student Performance Analytics*") 