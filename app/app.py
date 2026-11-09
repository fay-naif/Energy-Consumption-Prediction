from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from features import make_features
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Energy Consumption | التنبؤ باستهلاك الطاقة',page_icon='⚡',layout='wide')
ROOT=Path(__file__).resolve().parents[1]
@st.cache_resource
def load_bundle():
    return joblib.load(ROOT/'models'/'energy_model.joblib')

st.title('⚡ التنبؤ باستهلاك طاقة الأجهزة المنزلية')
st.caption('نموذج تجريبي | توقع استهلاك الأجهزة خلال فترة 10 دقائق بوحدة واط-ساعة (Wh)')
st.warning('⚠️ هذا النموذج دُرّب على بيانات منزل واحد فقط، ولا يُعدّ أداة موثوقة للتنبؤ باستهلاك المنازل الأخرى. النتائج تقديرية وليست قراءة فعلية أو نصيحة لترشيد الطاقة.')
if not (ROOT/'models'/'energy_model.joblib').exists():
    st.error('ملف النموذج غير موجود. شغّل أولًا: python src/train_and_export.py')
    st.stop()
bundle=load_bundle()
features=bundle['features']; defaults=bundle['defaults']; model=bundle['model']
st.info('أدخلي قراءات الحساسات الحقيقية إن كانت متاحة. القيم الافتراضية مأخوذة من وسيط بيانات التدريب، وليست قياسات لمنزلك؛ لذلك لا تعتبري الناتج توقعًا شخصيًا إذا اعتمدتِ عليها.')
with st.form('prediction'):
    c1,c2=st.columns(2)
    with c1:
        dt=st.datetime_input('التاريخ والوقت',value=datetime(2016,3,15,18,0))
        lights=st.number_input('استهلاك الإضاءة (Wh خلال 10 دقائق)',min_value=0.,value=float(defaults['lights']),step=1.)
        tout=st.number_input('درجة الحرارة الخارجية T_out (°C)',value=float(defaults['T_out']),step=.5)
        rhout=st.number_input('الرطوبة الخارجية RH_out (%)',min_value=0.,max_value=100.,value=float(defaults['RH_out']),step=1.)
    with c2:
        press=st.number_input('الضغط الجوي Press_mm_hg (mmHg)',value=float(defaults['Press_mm_hg']),step=1.)
        wind=st.number_input('سرعة الرياح Windspeed',min_value=0.,value=float(defaults['Windspeed']),step=.5)
        visibility=st.number_input('الرؤية Visibility',min_value=0.,value=float(defaults['Visibility']),step=1.)
        tdew=st.number_input('درجة الندى Tdewpoint (°C)',value=float(defaults['Tdewpoint']),step=.5)
    st.subheader('قراءات الحساسات الداخلية')
    st.caption('T1–T9 درجات حرارة تسع مناطق في المنزل، وRH_1–RH_9 الرطوبة النسبية في المناطق نفسها.')
    values={}
    with st.expander('إدخال قراءات الحساسات الـ18 (مطلوبة للتوقع الدقيق من حيث اكتمال المدخلات)',expanded=True):
        cols=st.columns(3)
        for i in range(1,10):
            with cols[(i-1)%3]:
                values[f'T{i}']=st.number_input(f'T{i} (°C)',value=float(defaults[f'T{i}']),step=.5,key=f't{i}')
                values[f'RH_{i}']=st.number_input(f'RH_{i} (%)',min_value=0.,max_value=100.,value=float(defaults[f'RH_{i}']),step=1.,key=f'rh{i}')
    submitted=st.form_submit_button('⚡ احسب الاستهلاك المتوقع',type='primary',width='stretch')
if submitted:
    row={**values,'lights':lights,'T_out':tout,'RH_out':rhout,'Press_mm_hg':press,'Windspeed':wind,'Visibility':visibility,'Tdewpoint':tdew}
    row['date'] = dt
    X = make_features(pd.DataFrame([row])).reindex(columns=features)
    prediction=max(0.,float(model.predict(X)[0]))
    st.metric('الاستهلاك المتوقع خلال 10 دقائق',f'{prediction:,.1f} Wh')
    st.caption(f'ما يعادل تقريبًا {prediction/1000:.3f} kWh خلال فترة التوقع فقط، وليس استهلاك ساعة أو يوم كامل.')
    st.caption('إذا استخدمتِ القيم الافتراضية، فالنتيجة توضيحية فقط؛ وللتنبؤ بحالة واقعية يجب توفير جميع قراءات الحساسات المطلوبة.')
st.divider()
st.markdown('**عن النموذج:** Extra Trees (إكسترا تريز) | بيانات UCI Appliances Energy Prediction (يو سي آي أبلاينسز إنرجي بريديكشن) | منزل واحد | قياسات كل 10 دقائق.')
st.caption('التقييم الأساسي باستخدام تقسيم عشوائي 80/20. يوجد أيضًا اختبار زمني مستقل موثق في reports/chronological_metrics.json؛ وقد يختلف أداؤه عن التقسيم العشوائي.')
