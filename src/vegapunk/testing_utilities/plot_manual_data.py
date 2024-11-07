# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from ioput.plots import plot_xy_data, scatter_xy_data, save_figure
# =============================================================================
# Summary: Generate plot by providing data explicitly
# =============================================================================
def plot_avg_prediction_loss(save_dir, is_save_fig=False,
                             is_stdout_display=False):
    """Plot average prediction loss of multiple models.
    
    Parameters
    ----------
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    """
    # Set training data set sizes
    training_sizes = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set example
    example = '3'
    # Set example data
    if example == 'erroneous_von_mises_properties':
        # Set models labels
        models_labels = ('GRU', 'Hybrid (worst cand.)',
                         'Hybrid (best cand.)', 'Candidate (worst)',
                         'Candidate (best)')
        # Initialize models average prediction loss
        models_avg_predict_loss = {}
        # Set models average prediction loss
        models_avg_predict_loss['GRU'] = \
            [789840.062, 163945.734, 174982.516, 55882.1094, 33157.9766,
             23583.9043, 14188.707, 2268.23462, 731.69928]
        models_avg_predict_loss['Hybrid (worst cand.)'] = \
            [87211.1328, 46635.0664, 33203.5078, 18189.6543, 7997.9668,
             7598.12549, 3608.25098, 820.335022, 330.519562]
        models_avg_predict_loss['Hybrid (best cand.)'] = \
            [5289.27295, 3571.09009, 3482.17212, 1808.65771, 1354.48035,
             653.862671, 388.217255, 92.2604599, 39.0522423]
        models_avg_predict_loss['Candidate (worst)'] = \
            len(models_avg_predict_loss['GRU'])*[2.43422850e+06,]
        models_avg_predict_loss['Candidate (best)'] = \
            len(models_avg_predict_loss['GRU'])*[1.40105781e+05,]
        # Set axes limits
        y_lims = (None, 2*10**7)
    elif example == 'learning_von_mises_hardening':
        # Set models labels
        models_labels = ('GRU', 'Hybrid', 'Candidate')
        # Initialize models average prediction loss
        models_avg_predict_loss = {}
        # Set models average prediction loss
        models_avg_predict_loss['GRU'] = \
            [344412.344, 213267.188, 111589.875, 53791.375, 32541.502,
             22198.9668, 17533.1738, 2447.47217, 599.129028]
        models_avg_predict_loss['Hybrid'] = \
            [29293.6641, 9592.91211, 5925.93701, 4881.25781, 2361.27759,
             1783.32178, 1379.46692, 210.375671, 76.2688141]
        models_avg_predict_loss['Candidate'] = \
            len(models_avg_predict_loss['Hybrid'])*[1.13914746e+04,]
        # Set axes limits
        y_lims = (None, None)
    elif example == 'learning_drucker_prager_pressure_dependency':
        # Initialize models average prediction loss
        models_avg_predict_loss = {}
        # Set models average prediction loss
        models_avg_predict_loss['GRU'] = \
            [12339631.0, 13428719.0, 3992861.0, 2326316.5, 1208200.12,
             318718.031, 167748.719, 78753.625, 18856.5293]
        models_avg_predict_loss['Candidate'] = \
            len(models_avg_predict_loss['GRU'])*[3.80905880e+07,]
        models_avg_predict_loss['Elastic + GRU'] = \
            [12446975.0, 9109351.0, 9864875.0, 2269201.75, 1291941.38,
             597997.625, 256310.562, 102906.891, 37765.375]
        models_avg_predict_loss['GRU(Candidate)'] = \
            [7015342.0, 8552386.0, 3552193.0, 2452850.5, 1094305.12, \
             475847.031, 168111.578, 55085.2969, 19909.418]
        #models_avg_predict_loss['GRU(Candidate) + Elastic'] = []
        models_avg_predict_loss['Candidate + GRU'] = \
            [11653675.0, 8254314.0, 2934598.75, 1318326.75, 759762.0,
             382273.969, 187543.328, 106387.133, 34965.8164]
        models_avg_predict_loss['Pretrained GRU'] = \
            [6499816.0, 3452356.25, 1869395.5, 827514.75, 492228.531,
             303312.469, 111247.281, 55598.6211, 40194.7734]
        models_avg_predict_loss['GRU debug'] = \
            [12339631.0, 13428718.0, 3992861.5, 2326316.5, 1208200.0, 318718.031, 167748.703, 78753.625, 18856.5293]
        # Set models labels
        models_labels = tuple(models_avg_predict_loss.keys())
        # Set axes limits
        y_lims = (None, None)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif example == 'learning_drucker_prager_pressure_gt_random_paths':
        # Initialize models average prediction loss
        models_avg_predict_loss = {}
        # Set models average prediction loss
        models_avg_predict_loss['GRU'] = \
            [13870940.0, 9007205.0, 5319528.0, 1973160.0, 863731.625,
             461373.812, 145931.891, 49301.1797, 23645.373]
        models_avg_predict_loss['GRU (P-T 1 deg)'] = \
            [4899814.0, 2918744.25, 1458804.0, 733764.5, 284074.094,
             259635.719, 126427.789, 68463.4531, 26346.2383]
        models_avg_predict_loss['GRU (P-T 2 deg)'] = \
            [2487198.5, 1751936.5, 888933.688, 342973.438, 211054.625,
             212256.391, 90464.5625, 81033.5156, 21000.9883]
        models_avg_predict_loss['GRU (P-T 5 deg)'] = \
            [726792.125, 639121.0, 260417.438, 129299.492, 78152.9531,
             69564.6406, 53452.2891, 47279.0938, 34439.5547]
        models_avg_predict_loss['GRU (P-T 15 deg)'] = \
            [726158.875, 423639.531, 187266.531, 69993.2969, 45646.2812,
             80390.7344, 44943.5977, 31853.9648, 10679.709]
        models_avg_predict_loss['GRU (P-T 20 deg)'] = \
            [1290995.62, 673243.125, 305679.688, 137858.969, 84545.6484,
             96000.0234, 58669.7344, 32626.041, 10720.3613]
        # Set models labels
        models_labels = tuple(models_avg_predict_loss.keys())
        # Set axes limits
        y_lims = (None, None)
    elif example == 'learning_drucker_prager_pressure_gt_proportional_paths':
        # Initialize models average prediction loss
        models_avg_predict_loss = {}
        # Set models average prediction loss
        models_avg_predict_loss['GRU-ID'] = \
            [786203.625, 249222.906, 170195.531, 75977.8438, 19342.9414,
             6608.13867, 2013.54932, 467.592468, 237.218658]
        models_avg_predict_loss['GRU-OD'] = \
            [46013296.0, 42630248.0, 40251524.0, 36468508.0, 34414744.0,
             33057892.0, 32254798.0, 31705688.0, 31617504.0]
        models_avg_predict_loss['GRU-ID (P-T 5 deg)'] = \
            [236565.828, 46638.1172, 19563.6523, 9115.92676, 5672.52148,
             2648.69092, 445.634979, 99.8270264, 28.1899223]
        models_avg_predict_loss['GRU-OD (P-T 5 deg)'] = \
            [4536985.5, 8192907.5, 8150818.5, 7426902.0, 9957678.0, 9725594.0,
             8040834.0, 8346771.0, 12388397.0]
        # Set models labels
        models_labels = tuple(models_avg_predict_loss.keys())
        # Set axes limits
        y_lims = (None, 10**9)
    elif example == 'learning_drucker_prager_pressure_vanilla_gru_strain_to_stress':
        # Initialize models average prediction loss
        models_avg_predict_loss = {}
        # Set testing data set type
        testing_type = 'proportional'
        # Set models average prediction loss
        if testing_type == 'random':
            models_avg_predict_loss['GRU (random/random)'] = \
                [16141004.0, 9427416.0, 4324756.0, 2051160.0, 847940.75,
                 463889.0, 147569.562, 56883.668, 20285.7812]
            models_avg_predict_loss['GRU (prop./random)'] = \
                [46518884.0, 43344628.0, 40094360.0, 36210344.0, 34093336.0,
                 33297962.0, 32818040.0, 31788200.0, 31427788.0]
            models_avg_predict_loss['GRU (prop. (1 cycle)/random)'] = \
                [50606952.0, 41617876.0, 33475532.0, 28173780.0, 20444084.0,
                 20943756.0, 15984476.0, 13502465.0, 12952019.0]
            models_avg_predict_loss['GRU (prop. (2 cycle)/random)'] = \
                [29639404.0, 30172134.0, 26068952.0, 15384506.0, 13722783.0,
                 11315563.0, 5638240.0, 5749249.5, 5466056.0]
            models_avg_predict_loss['GRU (random-mre/random)'] = \
                [21003232.0, 15302731.0, 6194713.0, 2493773.0, 1027621.25,
                 340247.625, 145146.5, 50868.2109, 19847.5781]
        elif testing_type == 'proportional':
            models_avg_predict_loss['GRU (random/prop.)'] = \
                [15467481.0, 21120532.0, 12350305.0, 2896517.75, 1199166.5,
                466586.812, 177469.219, 101425.297, 38876.7734]
            models_avg_predict_loss['GRU (prop./prop.)'] = \
                [837612.25, 283099.125, 171145.438, 69458.8203, 21866.0059,
                 6506.17676, 2028.60461, 474.383148, 67.6664429]
            models_avg_predict_loss['GRU (prop. (1 cycle)/prop.)'] = \
                [3746762.0, 593120.375, 449863.938, 203949.375, 69689.5938,
                 35576.2734, 17040.2109, 3632.01099, 933.311035]
            models_avg_predict_loss['GRU (prop. (2 cycle)/prop.)'] = \
                [2506393.0, 1967022.88, 1131010.25, 688241.875, 450861.688,
                 213495.891, 168481.281, 173237.047, 96353.1406]
            models_avg_predict_loss['GRU (random-mre/prop.)'] = \
                [31261064.0, 21174010.0, 12531494.0, 2553141.5, 757783.312,
                 316223.75, 122539.234, 72509.9297, 23320.418]
        # Set models labels
        models_labels = tuple(models_avg_predict_loss.keys())
        # Set axes limits
        y_lims = (None, 10**9)
    elif example == 'learning_drucker_prager_pressure_vanilla_gru_strain_to_pstrain':
        # Initialize models average prediction loss
        models_avg_predict_loss = {}
        # Set prediction type
        prediction_type = 'pstrain'
        # Set testing data set type
        testing_type = 'proportional'
        # Set models average prediction loss
        if testing_type == 'random':
            models_avg_predict_loss['GRU (random/random)'] = \
                [0.000366265071, 0.00020752789, 0.000132151559,
                 7.69877661e-05, 4.59879011e-05, 2.32560378e-05,
                 9.64250194e-06, 3.11591407e-06, 1.0348981e-06]
            models_avg_predict_loss['GRU (prop./random)'] = \
                [0.00072110293, 0.000596521772, 0.000538088032,
                 0.000504771015, 0.000491653103, 0.000499365386,
                 0.00049384078, 0.00049159216, 0.000477527443]
            models_avg_predict_loss['GRU (prop. (1 cycle)/random)'] = \
                [0.000656662975, 0.000561127206, 0.000478996837,
                 0.000447375933, 0.000364707666, 0.000232462073,
                 0.000203952703, 0.000199935763, 0.000171068605]
        elif testing_type == 'proportional':
            models_avg_predict_loss['GRU (random/prop.)'] = \
                [0.000184334873, 0.000153551882, 9.41468461e-05,
                 4.74615445e-05, 1.91845793e-05, 1.32747718e-05,
                 5.77544279e-06, 2.05993888e-06, 6.18703098e-07]
            models_avg_predict_loss['GRU (prop./prop.)'] = \
                [4.35151414e-05, 1.85219396e-05, 4.84384873e-06,
                 1.51599602e-06, 1.03068055e-06, 2.68887106e-07,
                 9.52368282e-08, 4.76807287e-08, 1.1902106e-08]
            models_avg_predict_loss['GRU (prop. (1 cycle)/prop.)'] = \
                [3.90881505e-05, 2.42470378e-05, 1.22666061e-05,
                 8.48056516e-06, 5.97190774e-06, 1.62412061e-06,
                 6.50477773e-07, 2.74945137e-07, 6.52306937e-08]
        # Set models labels
        models_labels = tuple(models_avg_predict_loss.keys())
        # Set axes limits
        y_lims = (None, 5*10**-3)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif example == '1':
        # Purpose
        # -------
        # Comparison of different types of strain-stress training data sets
        # in the prediction performance of the GRU model.
        #
        # Input features: Strain
        # Output features: Stress
        #
        # Training is performed with MRE. 
        #
        # Observations
        # ------------
        # (1) ...
        #
        # Data directories
        # ----------------
        # '/home/bernardoferreira/Documents/brown/projects/darpa_project/'
        # '7_local_hybrid_training/case_learning_drucker_prager_pressure/'
        # '2_vanilla_gru_model/strain_to_stress/mean_relative_error'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize models average prediction loss
        models_avg_predict_loss = {}
        # Set testing data set type
        testing_type = 'random'
        # Set prediction loss type
        prediction_loss_type = 'mre'
        # Set models average prediction loss
        if testing_type == 'random':
            models_avg_predict_loss['GRU (rand./rand.)'] = \
                [0.711485147, 0.765105605, 0.504647255, 0.325015962, 0.20673649, 0.125683844, 0.0986325219, 0.0481931716, 0.0259366408]
            models_avg_predict_loss['GRU (prop./rand.)'] = \
                [3.76491642, 2.53799725, 2.16523647, 1.94805944, 1.84417129, 1.87977254, 1.88211465, 1.85847485, 1.8168788]
            models_avg_predict_loss['GRU (prop.(1c)/rand.)'] = \
                [2.75709581, 2.68315554, 1.95425224, 1.87394953, 1.8384099, 1.38017488, 1.29872882, 1.22348166, 1.05631208]
            models_avg_predict_loss['GRU (prop.(2c)/rand.)'] = \
                [2.34063721, 2.09915113, 1.88783622, 1.42225647, 1.06488407, 1.06585515, 0.826798916, 0.784138799, 0.820805132]
            models_avg_predict_loss['GRU (prop.(4c)/rand.)'] = \
                [2.12748408, 1.7914269, 1.40952301, 1.23647201, 1.22088623, 1.1212368, 0.975950897, 0.958403409, 0.864891052]
            models_avg_predict_loss['GRU (rand+prop./rand.)'] = \
                [1.19108558, 1.07213414, 0.846771359, 0.411654711, 0.306536347, 0.223055899, 0.163353294, 0.0947590619, 0.05163607]
            models_avg_predict_loss['GRU (apex+cone/rand.)'] = \
                [0.891771078, 0.855269194, 0.745420575, 0.365836501, 0.230215341, 0.164953217, 0.125772104, 0.0626440346, 0.0331411585]
        elif testing_type == 'proportional':
            models_avg_predict_loss['GRU (rand./prop.)'] = \
                [0.619509816, 0.619414806, 0.54364872, 0.271952868, 0.155367523, 0.111418873, 0.0862796232, 0.0527936295, 0.0283560455]
            models_avg_predict_loss['GRU (prop./prop.)'] = \
                [1.02984631, 0.402982414, 0.215880111, 0.103733808, 0.0368126594, 0.0184596218, 0.0116598159, 0.00496042613, 0.00243825512]
            models_avg_predict_loss['GRU (prop.(1c)/prop.)'] = \
                [0.89505744, 0.60656476, 0.291863382, 0.181031168, 0.138794556, 0.0703007057, 0.0488198064, 0.0202571638, 0.0122310538]
            models_avg_predict_loss['GRU (prop.(2c)/prop.)'] = \
                [0.811315417, 0.648340762, 0.466697842, 0.381364524, 0.271140575, 0.173272818, 0.149255753, 0.111266643, 0.0770416409]
            models_avg_predict_loss['GRU (prop.(4c)/prop.)'] = \
                [1.05753779, 0.685935497, 0.65915519, 0.602456212, 0.522201836, 0.444880009, 0.36271733, 0.308862269, 0.267519295]
            models_avg_predict_loss['GRU (rand+prop./prop.)'] = \
                [0.496268153, 0.323136866, 0.206813589, 0.0868700743, 0.0509016514, 0.0429093353, 0.0314695463, 0.0144617511, 0.00852774922]
            models_avg_predict_loss['GRU (apex+cone/prop.)'] = \
                [0.591750145, 0.751824141, 0.69582051, 0.20526579, 0.0964431763, 0.0667821318, 0.0579240546, 0.0335734487, 0.0251233708]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set models labels
        models_labels = tuple(models_avg_predict_loss.keys())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes limits
        y_lims = (0, None)
        # Set axes scale
        x_scale = 'log'
        y_scale = 'linear'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif example == '2':
        # Purpose
        # -------
        # Comparison of different types of strain-stress training data sets
        # in the prediction performance of the GRU model.
        #
        # Input features: Strain
        # Output features: Stress
        #
        # Training is performed with MRE.
        #
        # Observations
        # ------------
        # (1) ...
        #
        # Data directories
        # ----------------
        # '/home/bernardoferreira/Documents/brown/projects/darpa_project/'
        # '7_local_hybrid_training/case_learning_drucker_prager_pressure/'
        # '2_vanilla_gru_model/strain_to_stress/mean_squared_error'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize models average prediction loss
        models_avg_predict_loss = {}
        # Set testing data set type
        testing_type = 'proportional'
        # Set prediction loss type
        prediction_loss_type = 'mre'
        # Set models average prediction loss
        if testing_type == 'random':
            models_avg_predict_loss['GRU (rand./rand.)'] = \
                [0.812820852, 0.587485433, 0.443491518, 0.291301489, 0.210853368, 0.14810878, 0.0836802274, 0.0616261959, 0.0436731428]
            models_avg_predict_loss['GRU (prop./rand.)'] = \
                [3.60349751, 2.53664994, 2.213696, 1.98230839, 1.87162042, 1.90855265, 1.92225015, 1.90025699, 1.8511728]
            models_avg_predict_loss['GRU (prop.(1c)/rand.)'] = \
                [2.70646811, 2.65650964, 1.979635, 1.81245673, 1.54497385, 1.43919969, 1.21374202, 1.16716945, 1.12689328]
            models_avg_predict_loss['GRU (prop.(2c)/rand.)'] = \
                [2.18302441, 2.0706296, 1.83727825, 1.27567708, 1.16543937, 1.09258831, 0.778883815, 0.687327206, 0.66747582]
            models_avg_predict_loss['GRU (prop.(4c)/rand.)'] = \
                [2.01952338, 1.68606234, 1.37279439, 1.1912148, 1.09493208, 1.05402493, 0.984184206, 0.805098653, 0.829387188]
            models_avg_predict_loss['GRU (rand+prop./rand.)'] = \
                [1.22022104, 1.03229606, 0.734650373, 0.465062201, 0.331751347, 0.2130595, 0.201604217, 0.111680336, 0.0549122095]
        elif testing_type == 'proportional':
            models_avg_predict_loss['GRU (rand./prop.)'] = \
                [0.47764051, 0.503305197, 0.635718703, 0.264252245, 0.178308457, 0.119197063, 0.0882135257, 0.0577588938, 0.0377396047]
            models_avg_predict_loss['GRU (prop./prop.)'] = \
                [0.706242919, 0.335800886, 0.258002162, 0.152450413, 0.0758856013, 0.0393763483, 0.0211755354, 0.0115520936, 0.00554641755]
            models_avg_predict_loss['GRU (prop.(1c)/prop.)'] = \
                [0.892419755, 0.557816446, 0.350053161, 0.216126323, 0.152940691, 0.106238894, 0.0495353453, 0.0274114236, 0.0310977902]
            models_avg_predict_loss['GRU (prop.(2c)/prop.)'] = \
                [0.914533079, 0.634667814, 0.474065334, 0.399839282, 0.310304999, 0.254926145, 0.158970207, 0.146383703, 0.11394386]
            models_avg_predict_loss['GRU (prop.(4c)/prop.)'] = \
                [1.06378746, 0.662446022, 0.648264825, 0.579989552, 0.55824858, 0.501351237, 0.448679268, 0.361505747, 0.31125167]
            models_avg_predict_loss['GRU (rand+prop./prop.)'] = \
                [0.691402555, 0.672018349, 0.22589989, 0.159687743, 0.0853741467, 0.0556534864, 0.0639128909, 0.0446022972, 0.015853446]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set models labels
        models_labels = tuple(models_avg_predict_loss.keys())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes limits
        y_lims = (0, None)
        # Set axes scale
        x_scale = 'log'
        y_scale = 'linear'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif example == '3':
        # Purpose
        # -------
        # Comparison of different sets of input features in the prediction
        # performance of the GRU model.
        #
        # Input features: (i) Strain, (ii) Strain + I1 + I2
        # Output features: Stress
        #
        # Training is performed with MRE.
        #
        # Observations
        # ------------
        # (1) ...
        #
        # Data directories
        # ----------------
        # '/home/bernardoferreira/Documents/brown/projects/darpa_project/'
        # '7_local_hybrid_training/case_learning_drucker_prager_pressure/'
        # '2_vanilla_gru_model/strain_to_stress/mean_relative_error'
        #
        # '/home/bernardoferreira/Documents/brown/projects/darpa_project/'
        # '7_local_hybrid_training/case_learning_drucker_prager_pressure/'
        # '2_vanilla_gru_model/strain_i1_i2_to_stress/mean_relative_error'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize models average prediction loss
        models_avg_predict_loss = {}
        # Set testing data set type
        testing_type = 'proportional'
        # Set prediction loss type
        prediction_loss_type = 'mre'
        # Set models average prediction loss
        if testing_type == 'random':
            models_avg_predict_loss['GRU (str, rand./rand.)'] = \
                [0.711485147, 0.765105605, 0.504647255, 0.325015962, 0.20673649, 0.125683844, 0.0986325219, 0.0481931716, 0.0259366408]
            models_avg_predict_loss['GRU (str, prop./rand.)'] = \
                [3.76491642, 2.53799725, 2.16523647, 1.94805944, 1.84417129, 1.87977254, 1.88211465, 1.85847485, 1.8168788]
            models_avg_predict_loss['GRU (str, prop.(2c)/rand.)'] = \
                [2.34063721, 2.09915113, 1.88783622, 1.42225647, 1.06488407, 1.06585515, 0.826798916, 0.784138799, 0.820805132]
            models_avg_predict_loss['GRU (str, rand+prop./rand.)'] = \
                [1.19108558, 1.07213414, 0.846771359, 0.411654711, 0.306536347, 0.223055899, 0.163353294, 0.0947590619, 0.05163607]
            models_avg_predict_loss['GRU (str, apex+cone/rand.)'] = \
                [0.891771078, 0.855269194, 0.745420575, 0.365836501, 0.230215341, 0.164953217, 0.125772104, 0.0626440346, 0.0331411585]
            models_avg_predict_loss['GRU (str-i1-i2, rand./rand.)'] = \
                [0.713928342, 0.762041032, 0.479262948, 0.28303951, 0.19211553, 0.122327089, 0.088236779, 0.0471768752, 0.024342183]
            models_avg_predict_loss['GRU (str-i1-i2, prop./rand.)'] = \
                [3.37488747, 2.37237024, 2.09414387, 1.89461029, 1.84086609, 1.91685247, 1.93387151, 1.94355845, 1.91204762]
            models_avg_predict_loss['GRU (str-i1-i2, prop.(2c)/rand.)'] = \
                [2.22479415, 1.96836734, 1.67915535, 1.33447719, 1.12187386, 1.32005167, 1.00056148, 0.976038933, 1.17263389]
            models_avg_predict_loss['GRU (str-i1-i2, rand+prop./rand.)'] = \
                [1.14428329, 0.764982581, 0.623999357, 0.382480979, 0.268026024, 0.185381353, 0.124198444, 0.0868264809, 0.0573462807]
            models_avg_predict_loss['GRU (str-i1-i2, apex+cone/rand.)'] = \
                [0.870067, 0.661266088, 0.417905986, 0.274378359, 0.213640466, 0.145024598, 0.107493408, 0.0576950461, 0.030576786]
        elif testing_type == 'proportional':
            models_avg_predict_loss['GRU (str, rand./prop.)'] = \
                [0.619509816, 0.619414806, 0.54364872, 0.271952868, 0.155367523, 0.111418873, 0.0862796232, 0.0527936295, 0.0283560455]
            models_avg_predict_loss['GRU (str, prop./prop.)'] = \
                [1.02984631, 0.402982414, 0.215880111, 0.103733808, 0.0368126594, 0.0184596218, 0.0116598159, 0.00496042613, 0.00243825512]
            models_avg_predict_loss['GRU (str, prop.(2c)/prop.)'] = \
                [0.811315417, 0.648340762, 0.466697842, 0.381364524, 0.271140575, 0.173272818, 0.149255753, 0.111266643, 0.0770416409]
            models_avg_predict_loss['GRU (str, rand+prop./prop.)'] = \
                [0.496268153, 0.323136866, 0.206813589, 0.0868700743, 0.0509016514, 0.0429093353, 0.0314695463, 0.0144617511, 0.00852774922]
            models_avg_predict_loss['GRU (str, apex+cone/prop.)'] = \
                [0.591750145, 0.751824141, 0.69582051, 0.20526579, 0.0964431763, 0.0667821318, 0.0579240546, 0.0335734487, 0.0251233708]
            models_avg_predict_loss['GRU (str-i1-i2, rand./prop.)'] = \
                [0.633915186, 0.62495482, 0.448864818, 0.208711296, 0.154933721, 0.115883753, 0.0777116865, 0.0421762504, 0.0292074382]
            models_avg_predict_loss['GRU (str-i1-i2, prop./prop.)'] = \
                [0.588817835, 0.325135231, 0.152866751, 0.0678516775, 0.0275869891, 0.0125249466, 0.00678915437, 0.00332685048, 0.00193638308]
            models_avg_predict_loss['GRU (str-i1-i2, prop.(2c)/prop.)'] = \
                [0.848101616, 0.62812376, 0.406069279, 0.333647847, 0.295855969, 0.221067965, 0.169398353, 0.125306949, 0.0872977823]
            models_avg_predict_loss['GRU (str-i1-i2, rand+prop./prop.)'] = \
                [0.422883153, 0.267492115, 0.160600051, 0.0821137354, 0.0447838083, 0.0272434261, 0.0137507729, 0.0129402373, 0.0089170821]
            models_avg_predict_loss['GRU (str-i1-i2, apex+cone/prop.)'] = \
                [0.629388154, 0.36820522, 0.206587791, 0.154914498, 0.0885878429, 0.058970768, 0.0411365405, 0.0220039301, 0.0203967094]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set models labels
        models_labels = tuple(models_avg_predict_loss.keys())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes limits
        x_lims = (None, None)
        y_lims = (0, None)
        # Set axes scale
        x_scale = 'log'
        y_scale = 'linear'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # Get number of training data set sizes
    n_training_sizes = len(training_sizes)
    # Get number of models
    n_models = len(models_labels)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array
    data_xy = np.full((n_training_sizes, 2*n_models), fill_value=None)
    # Loop over models
    for i, model_label in enumerate(models_labels):
        # Assemble model training data set size and average prediction loss
        data_xy[:, 2*i] = training_sizes
        data_xy[:, 2*i+1] = models_avg_predict_loss[model_label]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data labels
    data_labels = [x for x in models_avg_predict_loss.keys()]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Training data set size'
    y_label = f'Avg. prediction loss ({prediction_loss_type.upper()})'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot data
    figure, axes = plot_xy_data(
        data_xy, data_labels=data_labels, x_label=x_label, y_label=y_label,
        x_lims=x_lims, y_lims=y_lims, x_scale=x_scale, y_scale=y_scale,
        marker='o', markersize=3, markeredgecolor='k', markeredgewidth=0.5,
        is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Override plot legend (relocate)
    axes.legend(loc='upper right', ncols=1, frameon=True, fancybox=True,
                facecolor='inherit', edgecolor='inherit',
                fontsize=6, framealpha=1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        filename = 'testing_loss_convergence'
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close('all')
# =============================================================================
def plot_hardening_laws(save_dir, is_save_fig=False, is_stdout_display=False):
    """Plot hardening laws.
    
    Parameters
    ----------
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    """
    # Set accumulated plastic strain bounds
    acc_p_strain_min = 0.0
    acc_p_strain_max = 1.0
    # Set yield stress bounds (plot limits)
    yield_stress_min = None
    yield_stress_max = 1800
    # Set number of discretization points
    n_point = 200
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hardening laws labels
    hardening_labels = ('Ground-truth', 'Candidate (worst)',
                        'Candidate (best)')
    # Set accumulated plastic discrete points
    acc_p_strain = np.linspace(acc_p_strain_min, acc_p_strain_max, n_point)
    # Initialize hardening laws
    hardening_laws = {}
    # Set hardening laws
    hardening_laws['Ground-truth'] = \
        900 + 700*((acc_p_strain + 1e-5)**0.5)
    hardening_laws['Candidate (worst)'] = \
        400 + 300*acc_p_strain
    hardening_laws['Candidate (best)'] = \
        700 + 600*((acc_p_strain + 1e-5)**0.5)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # Get number of hardening laws
    n_laws = len(hardening_labels)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array
    data_xy = np.full((n_point, 2*n_laws), fill_value=None)
    # Loop over models
    for i, hardening_label in enumerate(hardening_labels):
        # Assemble model training data set size and average prediction loss
        data_xy[:, 2*i] = acc_p_strain
        data_xy[:, 2*i+1] = hardening_laws[hardening_label]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data labels
    data_labels = [x for x in hardening_laws.keys()]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Accumulated plastic strain'
    y_label = 'Yield stress (MPa)'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot data
    figure, axes = plot_xy_data(
        data_xy, data_labels=data_labels, x_label=x_label, y_label=y_label,
        x_lims=(acc_p_strain_min, acc_p_strain_max),
        y_lims=(yield_stress_min, yield_stress_max), x_scale='linear',
        y_scale='linear', is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Override plot legend (relocate)
    axes.legend(loc='upper left', ncols=1, frameon=True, fancybox=True,
                facecolor='inherit', edgecolor='inherit',
                fontsize=8, framealpha=1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        filename = 'hardening_laws'
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close('all')
# =============================================================================
if __name__ == "__main__":
    # Set plot processes
    is_plot_avg_prediction_loss = True
    is_plot_hardening_laws = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set plots directory
    plots_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                 'darpa_project/7_local_hybrid_training/'
                 'case_learning_drucker_prager_pressure/2_vanilla_gru_model/'
                 'strain_i1_i2_to_stress/mean_relative_error/plots')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot average prediction loss
    if is_plot_avg_prediction_loss:
        plot_avg_prediction_loss(save_dir=plots_dir, is_save_fig=True,
                                 is_stdout_display=True)
    # Plot hardening laws
    if is_plot_hardening_laws:
        plot_hardening_laws(save_dir=plots_dir, is_save_fig=True,
                            is_stdout_display=True)