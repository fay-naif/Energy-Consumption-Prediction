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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600;700&family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"], .stApp, input, button, label { font-family: 'IBM Plex Sans Arabic', 'Inter', sans-serif !important; }
.stApp {
    direction: rtl;
    color: #e8eef5;
    background:
        radial-gradient(1100px 600px at 90% -10%, #123a4a 0%, transparent 60%),
        radial-gradient(900px 600px at 0% 100%, #2a1d3f 0%, transparent 60%),
        #0a1220;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1250px; padding-top: 2.5rem; }
p, label, .stMarkdown, [data-testid="stCaptionContainer"] { text-align: right; }
.hero h1 { font-size: 2.3rem; font-weight: 700; margin: 0; color: #e8eef5; }
.hero h1 b { background: linear-gradient(90deg, #2ee6c5, #ffc65c); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { color: #8fa3b8; font-size: 1.05rem; margin-top: .4rem; }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stForm"] {
    background: linear-gradient(160deg, #111f33, #0c1726);
    border: 1px solid #1d324b; border-radius: 22px; padding: 1.6rem 1.8rem;
}
[data-testid="stNumberInput"] input, [data-testid="stDateInput"] input, [data-testid="stTimeInput"] input,
div[data-baseweb="input"], div[data-baseweb="select"] > div {
    background: #0a1524 !important; color: #e8eef5 !important; border-color: #22384f !important; border-radius: 12px !important;
}
[data-testid="stNumberInput"] button { background: #0a1524 !important; color: #2ee6c5 !important; border: none !important; }
label, [data-testid="stWidgetLabel"] p { color: #b6c4d3 !important; }
h2, h3, [data-testid="stHeading"] { color: #e8eef5 !important; text-align: right; }
[data-testid="stExpander"] { border: 1px solid #22384f; border-radius: 14px; background: rgba(10,21,36,.6); }
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #2ee6c5, #ffc65c) !important; color: #08131f !important;
    border: none !important; border-radius: 14px !important; font-weight: 700 !important; font-size: 1.1rem !important;
    padding: .8rem 0 !important; box-shadow: 0 8px 30px rgba(46,230,197,.25);
}
.result {
    display: flex; align-items: center; gap: 2rem; margin-top: 1.4rem;
    background: linear-gradient(160deg, #111f33, #0c1726); border: 1px solid #1d324b; border-radius: 22px; padding: 1.6rem 1.8rem;
}
.ring { width: 180px; height: 180px; border-radius: 50%; display: grid; place-items: center; flex: none; }
.ring div { width: 146px; height: 146px; border-radius: 50%; background: #0d1828; display: grid; place-items: center; text-align: center; }
.ring b { font-family: Inter; font-size: 2.2rem; color: #e8eef5; display: block; }
.ring small { color: #8fa3b8; }
.result .lbl { color: #8fa3b8; font-size: 1rem; }
.result .val { font-family: Inter; font-size: 2.6rem; font-weight: 700; color: #ffc65c; direction: ltr; text-align: right; }
.result .sub { color: #8fa3b8; font-size: .95rem; line-height: 1.8; }
hr { border-color: #1d324b !important; }
</style>
""", unsafe_allow_html=True)
ROOT=Path(__file__).resolve().parents[1]
@st.cache_resource
def load_bundle():
    return joblib.load(ROOT/'models'/'energy_model.joblib')

st.markdown("<div class='hero'><h1>⚡ التنبؤ <b>باستهلاك طاقة الأجهزة المنزلية</b></h1>"
            "<p>نموذج تجريبي | توقع استهلاك الأجهزة خلال فترة 10 دقائق بوحدة واط-ساعة (Wh)</p></div>", unsafe_allow_html=True)
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
    deg = min(prediction / 300, 1) * 360
    st.markdown(f"""
    <div class='result'>
      <div class='ring' style='background: conic-gradient(#ffc65c 0 {deg:.0f}deg, #1b2c42 0);'>
        <div><span><b>{prediction:,.1f}</b><small>Wh</small></span></div>
      </div>
      <div>
        <div class='lbl'>الاستهلاك المتوقع خلال 10 دقائق</div>
        <div class='val'>{prediction:,.1f} Wh</div>
        <div class='sub'>ما يعادل تقريبًا {prediction/1000:.3f} kWh خلال فترة التوقع فقط، وليس استهلاك ساعة أو يوم كامل.<br>
        إذا استخدمتِ القيم الافتراضية، فالنتيجة توضيحية فقط؛ وللتنبؤ بحالة واقعية يجب توفير جميع قراءات الحساسات المطلوبة.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
st.divider()
st.markdown('**عن النموذج:** Extra Trees (إكسترا تريز) | بيانات UCI Appliances Energy Prediction (يو سي آي أبلاينسز إنرجي بريديكشن) | منزل واحد | قياسات كل 10 دقائق.')
st.caption('التقييم الأساسي باستخدام تقسيم عشوائي 80/20. يوجد أيضًا اختبار زمني مستقل موثق في reports/chronological_metrics.json؛ وقد يختلف أداؤه عن التقسيم العشوائي.')
