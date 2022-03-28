import streamlit as st

st.write('''
Hello, StreamLit!
''')

ans = st.text_area('Say hello:')

st.write(f'''
{ans}
''')
