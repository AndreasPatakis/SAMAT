import pandas as pd
from sklearn.preprocessing import StandardScaler
from helper import train_eval_embanded_multi, train_eval_xgboost_multi, format_results, plot_results


# All partitions - groupings studied
def get_groupings(sign, dnn, symb, lyrical, X_train):
    partitions = {}

    sign = [col.lower().replace("_", "-").replace('-mean','').replace(' ','-') for col in sign]
    dnn = [col.lower().replace("_", "-").replace('-mean','').replace(' ','-') for col in dnn]
    symb = [col.lower().replace("_", "-").replace('-mean','').replace(' ','-') for col in symb]
    lyrical = [col.lower().replace("_", "-").replace('-mean','').replace(' ','-') for col in lyrical]

    sign = [k for k in sign if k in list(X_train.columns)]
    dnn = [k for k in dnn if k in list(X_train.columns)]
    symb = [k for k in symb if k in list(X_train.columns)]
    lyrical = [k for k in lyrical if k in list(X_train.columns)]

    # not included: attack-slope
    brightness = ['Spectral_spread_mean','Spectral_centroid_mean','Spectral_rolloff_mean','Zerocr_mean','Loudness','Pitch_salience_mean','Spectral_rms_mean','Spectral_flux_middle_high_mean','Spectral_flux_high_mean']

    # not: beats-loud
    danceability = ['Danceability','Onset-Rate','Rhythm Stability','Pulse_clarity_middle_low_mean','Pulse_clarity_low_mean','Pulse_clarity_mean','Pulse_clarity_middle_mean','Pulse_clarity_middle_high_mean','Pulse_clarity_high_mean']

    #not:
    tension = ['Dynamic-Complexity','Spectral_entropy_mean','Spectral_complexity_mean','Chords-Changes-Rate','Dissonance','Chords-Number-Rate','dom','Articulation','Rhythm Complexity','BPM','glob','mixolydian']

    # not: tonic?
    smoothness = ['Spectral_decrease_mean','Atonality','Mode','Melody','sub','Spectral-Energyband-High','Spectral-Energyband-Low', 'Spectral-Energyband-Middle-High',
                'Spectral-Energyband-Middle-Low','Spectral_flatness_high_mean','Spectral_flatness_middle_mean','Spectral_flatness_mean','Spectral_flatness_low_mean',
                'Spectral_flatness_middle_low_mean','Spectral_flatness_middle_high_mean','Spectral_kurtosis_mean','Attack_time_middle_low_mean','Attack_time_low_mean',
                'Attack_time_high_mean','Attack_time','Attack_time_middle_high_mean','Attack_time_middle_mean','Spectral_flux_low_mean',
                'Spectral_flux_middle_low_mean','Spectral_flux_mean','Spectral_flux_middle_mean','aelian','minor_fourth']

    spectral = ['Spectral_spread_mean','Spectral_centroid_mean','Spectral_rolloff_mean','Spectral_rms_mean','spectral-flux','spectral-flatness',
                'spectral-kurtosis','spectral-decrease','spectral-entropy','spectral-complexity']

    # tonic
    harmonic = ['atonality','mode','pitch-salience','melody','sub','dom','mixolydian','aelian','dissonance','glob','minor-fourth']

    # beats-loud
    rhythmic = ['danceability','pulse-clarity','rhythm-complexity','rhythm-stability']

    # attack-slope
    sound_shaping = ['dynamic-complexity','chords-changes-rate','chords-number-rate','attack-time','zerocr','articulation','onset-rate','loudness']

    brightness = [col.lower().replace("_", "-").replace('-mean','').replace(' ','-') for col in brightness]
    danceability = [col.lower().replace("_", "-").replace('-mean','').replace(' ','-') for col in danceability]
    tension = [col.lower().replace("_", "-").replace('-mean','').replace(' ','-') for col in tension]
    smoothness = [col.lower().replace("_", "-").replace('-mean','').replace(' ','-') for col in smoothness]

    spectral = [col.lower().replace("_", "-").replace('-mean','').replace(' ','-') for col in spectral]
    rhythmic = [col.lower().replace("_", "-").replace('-mean','').replace(' ','-') for col in rhythmic]
    harmonic = [col.lower().replace("_", "-").replace('-mean','').replace(' ','-') for col in harmonic]
    sound_shaping = [col.lower().replace("_", "-").replace('-mean','').replace(' ','-') for col in sound_shaping]

    partitions['original'] = {'sign': sign,
                        'dnn':dnn,
                        'symb':symb}

    partitions['with_lyrics'] = {
                        'sign': sign,
                        'dnn':dnn,
                        'symb':symb,
                        'lyrical':lyrical
    }

    partitions['approach1'] = {'Brightness, sharpness': brightness,
                        'Danceability, rhythm':danceability,
                        'Tension, complexity':tension,
                        'Acoustic smoothness':smoothness}

    partitions['approach1_lyrics'] = {'Brightness, sharpness': brightness,
                        'Danceability, rhythm':danceability,
                        'Tension, complexity':tension,
                        'Acoustic smoothness':smoothness,
                        'Lyrical': lyrical}

    partitions['approach2'] = {'Spectral': spectral,
                        'Harmonic':harmonic,
                        'Rhythmic':rhythmic,
                        'Sound Shaping':sound_shaping}

    partitions['approach2_lyrics'] = {'Spectral': spectral,
                        'Harmonic':harmonic,
                        'Rhythmic':rhythmic,
                        'Sound Shaping':sound_shaping,
                        'Lyrical': lyrical}

    # partitions['random'] = {
    #     '1':split_lists[0],
    #     '2':split_lists[1],
    #     '3':split_lists[2],
    #     '4':split_lists[3],
    #     '5':split_lists[4],
    # }

    return partitions


if __name__ == '__main__':
   # Read intial
    dataset = 'jamendo'

   # midlevel
    with open("../data/Jamendo/train-val-test/extras/midlevel.txt", "r", encoding="utf-8") as file:
        dnn = file.read().splitlines()  # Removes the newline characters

    # harmonic
    with open("../data/Jamendo/train-val-test/extras/harmonic.txt", "r", encoding="utf-8") as file:
        symb = file.read().splitlines()  # Removes the newline characters

    # signal
    with open("../data/Jamendo/train-val-test/extras/signal.txt", "r", encoding="utf-8") as file:
        sign = file.read().splitlines()  # Removes the newline characters

    # lyrical
    with open("../data/Jamendo/train-val-test/extras/lyrical.txt", "r", encoding="utf-8") as file:
        lyrical = file.read().splitlines()  # Removes the newline characters

    # labels
    with open("../data/Jamendo/train-val-test/extras/labels.txt", "r", encoding="utf-8") as file:
        labels = file.read().splitlines()  # Removes the newline characters

    # Read Splits
    tabular_path = '../data/Jamendo/train-val-test/tabular/'

    # Training data
    X_train = pd.read_pickle(tabular_path+'X_train.pkl')
    X_val = pd.read_pickle(tabular_path+'X_val.pkl')
    X_test = pd.read_pickle(tabular_path+'X_test.pkl')

    # Labels
    y_train = pd.read_pickle(tabular_path+'y_train.pkl')
    y_val = pd.read_pickle(tabular_path+'y_val.pkl')
    y_test = pd.read_pickle(tabular_path+'y_test.pkl')

    df_all = pd.read_pickle('../data/Jamendo/train-val-test/extras/df.pkl')

    X_train = pd.concat([X_train, X_val], ignore_index=True)
    y_train = pd.concat([y_train, y_val], ignore_index=True)
    all_feats = list(X_train.columns)

    # Normalize data
    scaler = StandardScaler()
    scaler.fit(X_train)

    scaled_train = scaler.transform(X_train)
    scaled_test = scaler.transform(X_test)

    X_train = pd.DataFrame(scaled_train, columns=X_train.columns)
    X_test = pd.DataFrame(scaled_test, columns=X_test.columns)

    # Get groupings
    partitions = get_groupings(sign, dnn, symb, lyrical, X_train)

    # Set up experiments
    approach = 'all_feats'
    if approach == 'all_feats':
        sign.remove('dissonance')
        dnn.remove('rhythm-stability')

    # Perform renames
    X_train.columns = X_train.columns.str.lower().str.replace("_", "-").str.replace('-mean','').str.replace(' ','-')
    X_test.columns = X_test.columns.str.lower().str.replace("_", "-").str.replace('-mean','').str.replace(' ','-')

    # Set results table
    labels_dict = {i:l for i,l in enumerate(labels)}
    results = pd.DataFrame(columns=['model','partition','importance','test','train'])

    # EMBanded
    model_name = 'embanded'
    for partition_name in partitions.keys():
        print(partition_name)
        results = train_eval_embanded_multi(X_train, y_train, X_test, y_test, partitions, partition_name, labels_dict, model_name, results)

    # XGBoost
    model_name = 'xgboost'
    for partition_name, partition in partitions.items():
        print(partition_name)
        results = train_eval_xgboost_multi(X_train, y_train, X_test, y_test, partition, partition_name, model_name, results)

    # Format results
    df, importance_df = format_results(results)
    
    # Plot results
    plot_results(df, importance_df, partitions, dataset, approach)

    # Save results
    df.to_pickle(f'{dataset}_results_importance_{approach}.pkl')
