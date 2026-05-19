import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import Pool
import plotly.graph_objects as go
import seaborn as sns
import joblib

st.set_page_config(
    page_title="Cargo Dwell Time Prediction",
    layout="wide"
)

sns.set_style("whitegrid")

model = joblib.load("cargo_dwell_time_catboost_model.pkl")
features_names = joblib.load("feature_names.pkl")
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
    margin-top: 0rem;
    max-width: 100%;
}
header{
    visibility: hidden;
}
.main.bloc`k-container {
    padding-top: 0rem;
}
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #12214d;
    margin-bottom: 0px;
}

.sub-title {
    font-size: 20px;
    color: #6c7893;
    margin-top: 0px;
    margin-bottom: 20px;
}

.blue-button button {
    background: linear-gradient(90deg,#0d6efd,#1f6cf0);
    color: white;
    height: 58px;
    border-radius: 15px;
    border: none;
    font-size: 22px;
    font-weight: 600;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#0d6efd,#1f6cf0);
    color: white;
    border: none;
    border-radius: 12px;
    height: 54px;
    font-size: 20px;
    font-weight: 600;
}

.metric-box {
    border: 1px solid #e4e8f2;
    border-radius: 15px;
    padding: 25px;
    height: 190px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background-color: white;
}

.prediction-number {
    font-size: 50px;
    color: #0d6efd;
    font-weight: 800;
    line-height: 1;
}

.hours-text {
    font-size: 28px;
    color: #0d6efd;
    font-weight: 700;
}
.excepted-release {
    font-size: 28px;
    color: #0d6efd;
    font-weight: 600;
    align-items: center;
    margin-top: 0px;
}


.small-title {
    font-size: 22px;
    font-weight: 700;
    color: #12214d;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Cargo Dwell Time Prediction System</div>', unsafe_allow_html=True)
tab1, tab2 = st.tabs([
    " Prediction",
    " Analytics Dashboard"
])

with tab1:
    col1, col2 = st.columns([1.1,0.9])

    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown('<div class="small-title">Shipment Information</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            date= st.date_input("Shipment Date",help="Arrival date of cargo shipment.")
            time= st.time_input("Shipment Time",help="Arrival time of cargo shipment.")

            cargo_weight = st.number_input("Cargo Weight (kg)", value=12500,help="Weight of the cargo in kilograms.")

            shipment_priority = st.selectbox(
                "Shipment Priority",
                ['Low','Medium','High'],
                    help="Priority level of the shipment."
                )
            shc = st.selectbox(
                "Special Handling Code (SHC)",
                [ 'PER','GEN', 'VAL', 'EAT', 'AVI', 'DGR'],help="Special Handling Code for cargo type."
            )
            

        with c2:
            customs_status = st.selectbox(
                "Customs Status",
                ['pending','Cleared','Under Inspection','Hold',],help="Current customs processing stage."
            )
            
            inspection_required = st.selectbox(
                "Inspection Required",
                ['Yes','No'],help="Whether shipment requires customs inspection."
            )
            clearance_type = st.selectbox(
                "Clearance Type",
                ['Manual','Automated'],help="Type of customs clearance process."
            )
            documentation_status = st.selectbox(
                "Documentation Status", 
                    ['Complete','Incomplete'],help="Status of shipment documentation."
            )

        with c3:
            payment_status = st.selectbox(
                "Payment Status",
                ['Paid','Pending','Overdue'],help="Current payment completion status."
            )

            consignee_name = st.selectbox(
                "Consignee Name",
                ['ABC Pharma Ltd','Global Electronics Pvt Ltd','DHL Logistics', 'Skyline Traders','FreshFoods Exports','CargoLink Pvt Ltd','Secure Freight Systems','AeroTech Solutions','MedSupply International','Prime Retail Imports'],help="Receiving company/customer name."           
            )
            consignee_behavior = st.selectbox(
                "Consignee Behavior",
                ['Average','Cooperative','Delayed Response'],help="Historical consignee release behavior."
            )
            staff_availability = st.number_input(
                "Staff Availability(%)",
                min_value=0,
                max_value=100,
                value=80,
                help="Available operational staff percentage."
            )


        with c4:
            truck_availability = st.selectbox(
                "Truck Availability",
                ['Available','Delayed','Not Available'],help="Availability of pickup truck for cargo."
            )

            packaging_condition = st.selectbox(
                "Packaging Condition",
                ['Good','Damaged','Needs Repacking'],help="Current packaging quality condition."
            )
            warehouse_load_percentage = st.slider(
                "Warehouse Load (%)",
                0,
                100,    
                75,
                help="Current warehouse occupancy percentage."
            )
            flight_delay_hours = st.slider(
                "Flight Delay (hours)",
                0.0,
                24.0,
                4.5,
                help="Flight delay affecting shipment arrival."
            )


        predict = st.button("🧳 Predict Dwell Time")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown('<div class="small-title">Prediction Results</div>', unsafe_allow_html=True)

        if predict:
            arrival_datetime = pd.to_datetime(
                str(date)
                + ' '
                + str(time)
            )
            arrival_month = arrival_datetime.month
            arrival_day = arrival_datetime.day
            arrival_hour = arrival_datetime.hour
            arrival_weekday = arrival_datetime.weekday()
            is_weekend = int(
                arrival_weekday in [5,6]
    )

            input_df = pd.DataFrame({
                'cargo_weight':[cargo_weight],
                'shipment_priority':[shipment_priority],
                'shc':[shc],
                'customs_status':[customs_status],
                'inspection_required':[1 if customs_status != 'Cleared' else 0],
                'clearance_type':['Manual'],
                'documentation_status':[documentation_status],
                'payment_status':[payment_status],
                'consignee_name':[consignee_name],
                'consignee_behavior':[consignee_behavior],
                'staff_availability':[staff_availability],
                'truck_availability':[truck_availability],
                'packaging_condition':[packaging_condition],
                'warehouse_load_percentage':[warehouse_load_percentage],
                'flight_delay_hours':[flight_delay_hours],
                'arrival_month':[arrival_month],
                'arrival_day':[arrival_day],
                'arrival_hour':[arrival_hour],
                'arrival_weekday':[arrival_weekday],
                'is_weekend':[is_weekend]

                
            })

            prediction = model.predict(input_df)[0]
            excepted_release=(arrival_datetime + pd.Timedelta(hours=prediction))
            metric1, metric2 = st.columns([1.2,1])
            with metric1:
                st.markdown(f'''
                <div class="metric-box">
                <div style="font-size:22px;font-weight:600;">Predicted Dwell Time</div>
                <div class="prediction-number">{round(prediction,1)}</div>
                <div class="hours-text">HOURS</div>
                </div>
                ''', unsafe_allow_html=True)

            with metric2:
                st.markdown(f'''
                <div class="metric-box">
                <div style="font-size:22px;font-weight:600;color:#2d2d2d;margin-top:25px;margin-bottom:10px;">Expected Date</div>
                <div class="excepted-release">
                {excepted_release.strftime("%d-%m-%Y")}
                <br>
                {excepted_release.strftime("%I:%M %p")}
                </div>
                ''', unsafe_allow_html=True)

            st.write("")

            st.markdown("### Top Delay Factors")
            input_pool = Pool(input_df,
                cat_features=[
                    'shipment_priority',
                    'shc',
                    'customs_status',
                    'clearance_type',
                    'documentation_status',
                    'payment_status',
                    'consignee_name',
                    'consignee_behavior',
                    'truck_availability',
                    'packaging_condition'
                ]
            )
            shap_values = model.get_feature_importance(input_pool,type='ShapValues')
            feature_impact = np.abs(
                shap_values[0][:-1]
            )
            importance_df = pd.DataFrame({"Feature": input_df.columns,"Importance": feature_impact})
            importance_df['Feature']=importance_df['Feature'].str.replace('_',' ').str.title()
            total_importance = importance_df["Importance"].sum()
            importance_df["Importance"] = (importance_df["Importance"]/total_importance)* 100
            importance_df = importance_df.sort_values(by="Importance", ascending=True).tail(7)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=importance_df["Importance"].round(1),
                    y=importance_df["Feature"],
                    orientation='h',
                    text=[
                        f"{round(x,1)}%"
                        for x in importance_df["Importance"]
                    ],
                    textposition='outside',
                    marker=dict(
                        color='#4c72b0'
                    )
                )
            )
            fig.update_layout(
                height=250,
                plot_bgcolor='#f5f7fb',
                paper_bgcolor='#f5f7fb',
                margin=dict(
                    l=90,
                    r=10,
                    t=10,
                    b=10
                ),

                xaxis=dict(
                    visible=False
                ),

                yaxis=dict(
                    title=''
                )
            )

            st.plotly_chart(

                fig,

                use_container_width=True,

                config={
                    "displayModeBar": False
                }
            )
        else:
            st.info("Enter values and click Predict Dwell Time")

        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown(
        "## Cargo Analytics Dashboard"
    )

    st.write("")

    full_data = pd.read_csv("cargo_dwell_full_dataset.csv")
    full_data['arrival_datetime'] = pd.to_datetime(full_data['arrival_datetime'])
    st.subheader("Date Range Filter")
    min_date = full_data['arrival_datetime'].min().date()
    max_date = full_data['arrival_datetime'].max().date()
    date_range = st.date_input(
        "Select Date Range",
        value=(min_date,max_date),
        min_value=min_date,
        max_value=max_date
    )
    filtered_df = full_data[
        (full_data['arrival_datetime'] >= pd.to_datetime(date_range[0])) &
        (full_data['arrival_datetime'] <= pd.to_datetime(date_range[1]))
    ]

    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )
        fig1, ax1 = plt.subplots(
            figsize=(5,3)
        )
        sns.histplot(
            filtered_df['dwell_time_hours'],
            bins=30,
            kde=True,
            color='#4c72b0',
            ax=ax1
        )
        ax1.set_title(
            '1. Distribution of Dwell Time'
        )
        ax1.set_xlabel(
            'Dwell Time Hours'
        )
        st.pyplot(fig1)
        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with row1_col2:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        truck_data = filtered_df.groupby(

            'truck_availability'

        )['dwell_time_hours'].mean().reset_index()

        fig2, ax2 = plt.subplots(
            figsize=(5,3)
        )

        sns.barplot(
            data=truck_data,
            x='truck_availability',
            y='dwell_time_hours',
            color='#4c72b0',
            ax=ax2
        )

        for i,v in enumerate(
            truck_data['dwell_time_hours']
        ):

            ax2.text(
                i,
                v+1,
                round(v,1)
            )

        ax2.set_title(
            '2. Truck Availability vs Dwell'
        )
        ax2.set_xlabel(
            'Truck Availability'
        )
        ax2.set_ylabel(
            'Dwell Time Hours'
        )

        st.pyplot(fig2)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with row1_col3:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        payment_data = filtered_df.groupby(

            'payment_status'

        )['dwell_time_hours'].mean().reset_index()

        fig3, ax3 = plt.subplots(
            figsize=(5,3)
        )

        sns.barplot(

            data=payment_data,

            x='payment_status',

            y='dwell_time_hours',

            color='#4c72b0',

            ax=ax3
        )

        for i,v in enumerate(
            payment_data['dwell_time_hours']
        ):

            ax3.text(
                i,
                v+1,
                round(v,1)
            )

        ax3.set_title(
            '3. Payment Status vs Dwell'
        )
        ax3.set_xlabel(
            'Payment Status'
        )
        ax3.set_ylabel(
            'Dwell Time Hours'
        )

        st.pyplot(fig3)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )
        filtered_df['release_datetime'] = pd.to_datetime(
            filtered_df['release_datetime']
        )
        month_order = [
            'Jan',
            'Feb',
            'Mar',
            'Apr',
            'May',
            'Jun',
            'Jul',
            'Aug',
            'Sep',
            'Oct',
            'Nov',
            'Dec'
        ]
        monthly_data = filtered_df.groupby(
            filtered_df['release_datetime'].dt.strftime('%b')
        )['dwell_time_hours'].mean().reindex(month_order).reset_index()
        monthly_data.columns = ['Month','Dwell']
        monthly_data = monthly_data.dropna()
        
        fig4, ax4 = plt.subplots(
            figsize=(5,3)
        )
        sns.lineplot(
            data=monthly_data,
            x='Month',
            y='Dwell',
            marker='o',
            color='#4c72b0',
            ax=ax4
        )

        for i,value in enumerate(
            monthly_data['Dwell']
        ):
            ax4.annotate(
            f"{value:.1f}",
            (i, value),
            textcoords="offset points",
            xytext=(0,8),
            ha='center',
            fontsize=9
    )

        ax4.set_title(
            '4. Monthly Dwell Trend'
        )

        ax4.set_xlabel(
            'Month'
        )

        ax4.set_ylabel(
            'Dwell Time Hours'
        )

        st.pyplot(fig4)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with row2_col2:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        feature_importance = pd.DataFrame({
            'Feature': features_names,
            'Importance': model.feature_importances_
        })

        feature_importance = feature_importance.sort_values(

            by='Importance',

            ascending=False

        ).head(5)

        fig5, ax5 = plt.subplots(
            figsize=(3.5,3)
        )

        sns.barplot(

            data=feature_importance,

            x='Importance',

            y='Feature',

            color='#4c72b0',

            ax=ax5
        )

        for i,v in enumerate(
            feature_importance['Importance']
        ):

            ax5.text(
                v+0.2,
                i,
                round(v,1)
            )

        ax5.set_title(
            '5. Feature Importance'
        )

        st.pyplot(fig5)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with row2_col3:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )
        customs_data = filtered_df[
            'customs_status'
        ].value_counts()

        fig6, ax6 = plt.subplots(
            figsize=(3.5,3)
        )

        ax6.pie(
            customs_data.values,
            labels=customs_data.index,
            autopct='%1.1f%%',
            radius=0.8,
            colors=[
                '#4c72b0',
                '#89a2d9',
                '#b7c6e6'
            ]
        )
        ax6.set_aspect('0.6')
        ax6.set_title(
            '6. Customs Status Distribution'
        )

        st.pyplot(fig6)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )
