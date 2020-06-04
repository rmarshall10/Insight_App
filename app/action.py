from flask import render_template
from flask import request
from app import app
from app import video_tracker
from flask import send_from_directory
#from app.a_Model import ModelIt
import os
from werkzeug import secure_filename

@app.route('/')
@app.route('/index')
def index():
	return render_template("index.html", title = 'Home', user = { 'nickname': 'rockstar!' })

@app.route('/upload')
def upload_file():
	return render_template('upload.html')

def allowed_extension(filename):
	if not "." in filename:
		return False
	ext = filename.rsplit(".",1)[1]

	if ext.upper() in app.config['ALLOWED_EXTENSIONS']:
		return True
	else:
		return False
		
	
@app.route('/uploader', methods = ['GET', 'POST'])
def uploaded_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save(secure_filename(f.filename))
		#return 'file uploaded successfully'
		model_path = "app/static/"
		model_name = "frozen_inference_graph.pb"
		model_text = "graph_text.pbtxt"
		net = video_tracker.loading_model(model_path + model_name, model_path + model_text)
		
		bounces = video_tracker.run_video(f.filename, net)

		#print("--- %s seconds ---" % (time.time() - start_time))

		return str(bounces)

		#return render_template('video_output.html')

# cv2.setUseOptimized(True) Streamlit
# path = 'ball_test3.mp4'
# #path = "juggle_clip2.mp4"
# net = loading_model('frozen_inference_graph_sc_ball3.pb', 'graph_sc2.pbtxt')
# start_time = time.time()
# bounces = run_video(path, net)
# print(f"You got {bounces} juggles! Try to beat it next time!")

# print("--- %s seconds ---" % (time.time() - start_time))





# Python code to connect to Postgres
# You may need to modify this based on your OS, 
# as detailed in the postgres dev setup materials.
# user = 'ryanmarshall' #add your Postgres username here      
# host = 'localhost'
# dbname = 'birth_db'
# db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = user)



# @app.route('/db')
# def birth_page():
# 	sql_query = """                                                             
#                SELECT * FROM birth_data_table WHERE delivery_method='Cesarean'\
# ;                                                                               
#                """
# 	query_results = pd.read_sql_query(sql_query,con)
# 	births = ""
# 	print(query_results[:10])
# 	for i in range(0,10):
# 		births += query_results.iloc[i]['birth_month']
# 		births += "<br>"
# 	return births

# @app.route('/db_fancy')
# def cesareans_page_fancy():
# 	sql_query = """
#               SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
#                """
# 	query_results=pd.read_sql_query(sql_query,con)
# 	births = []
# 	for i in range(0,query_results.shape[0]):
# 		births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
# 	return render_template('cesareans.html',births=births)

# @app.route('/input')
# def juggler_input():
#    return render_template("input.html")




# @app.route('/output')
# def juggler_output():
# 	#pull 'birth_month' from input field and store it
# 	patient = request.args.get('birth_month')
# 	#just select the Cesareans  from the birth dtabase for the month that the user inputs
# 	query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
# 	print(query)
# 	query_results=pd.read_sql_query(query,con)
# 	print(query_results)
# 	births = []
# 	for i in range(0,query_results.shape[0]):
# 		births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
# 		the_result = ModelIt(patient,births)
# 	return render_template("output.html", births = births, the_result = the_result)