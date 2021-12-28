README
% Date: 20.10.2021 by Philip von Platen

The files contained in this folder each contain a single PEEP titration. 
The experiments were perfomed at the Charite, Berlin as part of the the BMBF Project SOLVe

The measured data was transmitted from the Fritz Stephan GmbH EVE respirator.
Data was was recorded using a dspace microlabbox and panel PC running MATLAB 2017b and ControlDesk ver 6.3.
All data is recorded at 100 Hz evenhtough sample times for various values (see below) were transmitted serially and hence only udpated at a slower rate.

The data saved in this folder has been preprocced. The given data is resampled on a breath by breath basis. Therefore each data point is the mean value of the data contained in that breath (unless stated otherwise as schown below).
Additionally the values C_rs_est, R_rs_est and p_peep_est are estimated on a breath-by-breath basis using a multiple least-squares regression (see Bates - Lung Mechanics An Inverse Modelling Approach). The coefficent of determination and confidence intervals are also saved.

Please note the following:
- the data is cropped perfectly and the start of the titration


Each file contains the following signals:

header		|	description 				| 	unit	|		update time		| 	comment
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
timestamp 		UNIX time stamp 				none			breath				none
t 			time since measurment start in 			sec			breath				none
p_peep			positive end expiratory pressure		mbar			breath				none	
p_pip           	peak insp pressure  	             		mbar			breath				none
p_plat          	plateau pressure 		                mbar			breath				none
p_aw_avg        	mean airawy pressure                    	mbar			breath				none
V_tidal         	tidal volume                            	ml			breath				none
f               	breathing frequency                     	breaths/min		breath				none
frac_insp_O2    	fraction inspired oxygen 	            	%			breath				this value is measured and lags behind the settings by about 50 - 100 msec
P_et_CO2		end-tidal partial pressure CO2			mmHg			breath				measured using capnography in the main stream (MASSIMO)
sat_p_O2        	oxygen saturation via pulse oximeter 		%			breath				measured at the tail of the animal. Not always a clean signal
puls_rate 		pulse rate from the SPO2 sensor			beats/min		breath				none
puls_index		pulsation idnex from the SPO2 sensor		none			breath				none	
C_rs_eve        	compliance estimate (respirator)        	ml/mbar			breath				internal estimate by the EVE respirator, average value over 5 breaths!
R_rs_eve        	resistance estimate (respirator)        	mbar/L/sec		breath				internal estimate by the EVE respirator, average value over 5 breaths!
R_rs_est        	resistance estimate                     	mbar/L/sec		breath				new/better estimate calculated for each breath
C_rs_est        	compliance estimate                     	mL/mbar			breath				new/better estimate calculated for each breath
p_peep_est      	PEEP estimate                     		mbar			breath				new/better estimate calculated for each breath
coef_determ		coeffcient of determination			none			breath				coefficient of determination as given by Bates - Lung Mechanics An Inverse Modelling Approach page 49
