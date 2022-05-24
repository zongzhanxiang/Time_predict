import streamlit as st
import pandas as pd

def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')




# layout='centered' 指定居中;展开sidebar
st.set_page_config(page_title='ISTFCGSE',
page_icon = '',layout='centered',initial_sidebar_state='expanded')

st.write("## ISTFCGSE : Infer when your data is sampled ")
#ISTFCGSE 即 Inferring sampling time from core gene set expression
st.write('<center>This is a machine learning application that infers sample sampling time points based on rhythmic core gene set expression levels.</center>',unsafe_allow_html=True)

menu = ['Home','Predict','About']
choice = st.sidebar.selectbox("Menu",menu)


if choice == 'Home':
	from PIL import Image
	img = Image.open("./data/fig1.jpeg")
	st.write()
	with st.expander("Background"):
		st.text("""
			The biological clock controls the expression of about 30% of genes in plants, 
			and is particularly important for plant growth and development. 
			Regulating the biological clock and its related components is an innovative 
			idea to achieve crop genetic improvement. The existence of large-scale 
			transcriptome sequencing data lays the foundation for intelligent regulation 
			of crop expression with the help of circadian clock. 
			However, most transcriptome data are sequenced without taking into account the 
			time point of recording. To this end, we developed this tool to predict the 
			sampling time points corresponding to plant transcriptomic data to activate 
			the circadian regulation potential of existing large-scale transcriptomic data.
			""")
	st.image(img,use_column_width=True)
	st.write('<center>From https://www.science.org/doi/10.1126/science.abc9141</center>',unsafe_allow_html=True)
	
elif choice == 'Predict':
	st.write('### Upload expression data for core gene sets for prediction')
	st.write(' ')

	Symblos_MSU7_RAP = pd.read_csv('./data/Symblos_MSU7_RAP.csv',sep='\t')
	Symblos_MSU7_RAP = Symblos_MSU7_RAP.sort_values(by='Symblos').reset_index(drop=True)

	Input_Example = pd.read_csv('./data/Input_Example.csv',sep='\t')
	#Input_Example = Input_Example.sort_values(by='Symblos').reset_index(drop=True)


	left_column, right_column = st.columns(2)
	left_column.write('You need to upload a CSV file in the following format, separated by TAB')
	left_column.dataframe(Input_Example,width=400)

	right_column.write('The relationship between Symbols, Loc number, and RAP number is provided here')
	right_column.dataframe(Symblos_MSU7_RAP,width=400)


	# 获取数据
	#data_file = None
	data_file = st.file_uploader('Upload Data',type=['csv'])
	if data_file is not None:
		Input = pd.read_csv(data_file,sep='\t')
		if Input is not None:
			from pycaret.classification import load_model, predict_model
			model = load_model('./data/lda_test')
			Input = Input.T
			Input.columns = Input.loc['Symblos']
			Input = Input.drop('Symblos',axis=0)

			st.write('')
			st.write('Results will be returned shortly below')
			predict_time = predict_model(model,Input)


			st.dataframe(pd.DataFrame(predict_time.iloc[:,-2:]))
			st.write('')
			st.write('Completion')
			csv = convert_df(pd.DataFrame(predict_time.iloc[:,-2:]))
			st.download_button(label="Download data as CSV",
                data=csv,
                file_name='result.csv',
                mime='text/csv')
elif choice == 'About':
	st.markdown('### <center>About our model</center>',unsafe_allow_html=True)

	st.markdown("""
		We train the model on GSE36040 data using the automated machine learning tool Pycaret. 
		The ratio of training set and test set is 7 to 3 (random seed 666), 
		and the accuracy rate of 10-fold cross-validation on the training set is 0.9617. 
	    The test set confusion matrix is shown below. 
		""",unsafe_allow_html=True)

	from PIL import Image
	img2 = Image.open("./data/fig2.png")
	st.write('')
	st.image(img2,use_column_width=True)
	st.write('')
	st.write("""Almost all samples can be matched to the exact time point, 
		and a few incorrectly matched time points are also close to the actual sampling time.""")