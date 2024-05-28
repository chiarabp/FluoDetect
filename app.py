from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from io import StringIO
from AddSpectra import AddSpectra
from DB import DB
from PeaksSpot import peaks_spot
import json
from openai import OpenAI
from CompareSpectra import CompareSpectra

client = OpenAI(
    # This is the default and can be omitted
     api_key='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'
)

app = Flask(__name__)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        account_exists = DB.login(username, password)
        if account_exists:
            return redirect(url_for(''))
        else:
            error = 'Invalid Credentials. Please try again.'

    return render_template('login.html', error=error)

@app.route('/', methods=['GET', 'POST'])
def compare_spectrums():
    if request.method == 'POST':
        # Handle form submission
        wavs_to_delete = [float(request.form['led_range_b']), float(request.form['led_range_e'])]
        smoothing_wind = int(request.form['smoothing_wind'])
        min_peak_height = float(request.form['min_peak_height'])
        alpha = int(request.form['alpha'])
        db_name = request.form['db_select']

        dbs_available_df, _ = DB.get_databases_list()
        dbs_available = dbs_available_df['name'].tolist()

        if 'load_spectrum' in request.form:
            load_spectrum = int(request.form['load_spectrum'])
        else:
            load_spectrum = 0

        if 'change_spectrum' in request.form:
            change_spectrum = int(request.form['change_spectrum'])
        else:
            change_spectrum = 0

        # means spectrum change button has been touched, reload the page without spectrum
        if change_spectrum == 1:
            options_selected = {
                'wavs_to_delete': wavs_to_delete,
                'smoothing_wind': smoothing_wind,
                'min_peak_height': min_peak_height,
                'alpha': alpha
            }
            return render_template('compare_spectrums.html', options=options_selected, plot_original=None,
                                   plot_elaborated=None)

        # 'spectrum' is a request.files, 'file_df' is a hidden request.form
        if 'spectrum' not in request.files and 'file_df' not in request.form:
            return render_template('compare_spectrums.html', error='No file part', options=None)
        # maybe it has been sent in the post request...
        elif 'file_df' in request.form:
            df_inserted = pd.read_json(StringIO(request.form['file_df']), orient='split')
            df_inserted.columns = ['Wavelength [nm]', "Intensity [counts]", "not interested1", "not interested2"]
            filename = request.form['filename']
        # must have been inserted now, read it and check if valid
        else:
            file = request.files['spectrum']
            # Check if the file is not empty
            if file.filename == '':
                return render_template('compare_spectrums.html', error='No selected file', options=None)
            filename = file.filename

            # Check if the file is allowed (optional)
            allowed_extensions = {'csv', 'txt'}
            if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                return render_template('compare_spectrums.html', error='Invalid file extension', options=None)

            # Read the file using Pandas
            try:
                if file.filename.endswith('.csv') or file.filename.endswith('.txt'):
                    df_inserted = pd.read_csv(file, sep=";", header=5)
                    df_inserted.columns = ['Wavelength [nm]', "Intensity [counts]", "not interested1", "not interested2"]
                else:
                    return render_template('compare_spectrums.html', error='Unsupported file format', options=None)
            except pd.errors.EmptyDataError:
                return render_template('compare_spectrums.html', error='File is empty', options=None)

        # Process data
        options_selected = {
            'wavs_to_delete': wavs_to_delete,
            'smoothing_wind': smoothing_wind,
            'min_peak_height': min_peak_height,
            'alpha': alpha,
            'file': filename,
            'dbs': dbs_available,
            'db_name': db_name
        }

        comp_set = {
            'pk_num_tolerance': float(request.form['pk_num_tolerance']),
            'pk_tolerance': float(request.form['pk_tolerance']),
            'fwhm_tolerance': float(request.form['fwhm_tolerance']),
            'weight_peak': float(request.form['weight_peak']),
            'weight_fwhm': float(request.form['weight_fwhm']),
            'weight_shape': float(request.form['weight_shape']),
            'num_records_to_show': int(request.form['num_records_to_show']),
            'shape_pt_tol': float(request.form['shape_pt_tol']),
            'baseline_pts': float(request.form['baseline_pts'])
        }

        if load_spectrum:
            add_spectra_class = AddSpectra(wavs_to_delete, smoothing_wind, min_peak_height, alpha, df_inserted)
            wavelengths_reduced, intensities_reduced, wavelengths, intensities, locations, fwhm, width_heights, left_ips, right_ips = add_spectra_class.analyze_spectra()

            df_reduced = pd.DataFrame(
                {'Wavelength [nm]': wavelengths_reduced, 'Intensity [counts]': intensities_reduced})
            df_data = pd.DataFrame({'Peak wavelengths [nm]': locations, 'FWHM peaks [nm]': fwhm})

            file_df = df_inserted.to_json(orient='split')

            plot_elaborated = create_plot(df_reduced, 'Inserted spectrum', 2, locations, width_heights, left_ips, right_ips)

            return render_template('compare_spectrums.html', file_df=file_df, options=options_selected,
                                   comp_set=comp_set, plot_inserted=plot_elaborated,
                                   table=df_data.to_html(
                                       classes='table table-striped table-bordered table-hover',
                                       index=False))

        if request.form['compare_spectrum'] == '1' and db_name == 'Choose DB':
            add_spectra_class = AddSpectra(wavs_to_delete, smoothing_wind, min_peak_height, alpha, df_inserted)
            wavelengths_reduced, intensities_reduced, wavelengths, intensities, locations, fwhm, width_heights, left_ips, right_ips = add_spectra_class.analyze_spectra()

            df_reduced = pd.DataFrame(
                {'Wavelength [nm]': wavelengths_reduced, 'Intensity [counts]': intensities_reduced})
            df_data = pd.DataFrame({'Peak wavelengths [nm]': "{:.2f}".format(locations), 'FWHM peaks [nm]': "{:.2f}".format(fwhm)})

            file_df = df_inserted.to_json(orient='split')

            plot_elaborated = create_plot(df_reduced, 'Inserted spectrum', 2, locations, width_heights, left_ips, right_ips)

            return render_template('compare_spectrums.html', error='Choose a DB',
                                   file_df=file_df, options=options_selected,
                                   comp_set=comp_set, plot_inserted=plot_elaborated,
                                   table=df_data.to_html(classes='table table-striped table-bordered table-hover',
                                       index=False))

        compare_spectra_class = CompareSpectra(db_name, wavs_to_delete, smoothing_wind, min_peak_height,
                                               alpha, df_inserted, comp_set)
        (comp_records, wavelengths_reduced, intensities_reduced, wavelengths, intensities, locations, fwhm,
         width_heights, left_ips, right_ips) = compare_spectra_class.compare_spectra()

        # show before the conditioned spectrum
        columns_plot = ['Wavelength [nm]', 'Intensity [counts]']
        columns_table = ['Peak wavelengths [nm]', 'FWHM peaks [nm]']

        df_inserted_reduced = pd.DataFrame({columns_plot[0]: wavelengths_reduced, columns_plot[1]: intensities_reduced})
        df_inserted_data = pd.DataFrame({columns_table[0]: [["{:.2f}".format(x) for x in locations]], columns_table[1]: [["{:.2f}".format(x) for x in fwhm]]})

        plot_inserted = create_plot(df_inserted_reduced, 'Inserted spectrum', 2, locations, width_heights, left_ips, right_ips)

        file_df = df_inserted.to_json(orient='split')

        # now lets extract pandas dataframes and plot from the comp_records
        df_comp_records = pd.DataFrame(columns=columns_plot)
        df_comp_records_data = pd.DataFrame(columns=['Name', 'Peak wavelengths [nm]', 'FWHM peaks [nm]', 'Score [%]'])
        min_peak_heights = []
        record_ids = []

        for record in comp_records:
            wavelengths_comp, intensities_comp = record.get_spectra_dim_reduction(alpha)
            df_comp_records.loc[len(df_comp_records)] = {'Wavelength [nm]': wavelengths_comp, 'Intensity [counts]': intensities_comp}
            df_comp_records_data.loc[len(df_comp_records_data)] = {'Peak wavelengths [nm]': record.pks_wav,
                                                                   'FWHM peaks [nm]': record.fwhm, 'Name': record.name,
                                                                   'Score [%]': record.score}
            min_peak_heights.append(record.min_peak_height)
            record_ids.append(record.id_)

        # create plots and array them
        plots_comp = []
        i_ = 0
        for index, row in df_comp_records.iterrows():
            # row[0] is the index
            locations_c, width_heights_c, left_ips_c, right_ips_c, _ = peaks_spot(row[columns_plot[0]], row[columns_plot[1]], min_peak_heights[i_])
            df_comp_params = {'locations': locations_c, 'width_heights': width_heights_c, 'left_ips': left_ips_c, 'right_ips': right_ips_c}
            plots_comp.append(create_plot(df_inserted_reduced, 'Inserted spectra', 3, locations=locations,
                                             width_heights=width_heights, left_ips=left_ips, right_ips=right_ips,
                                             df_comp=row, df_comp_name=df_comp_records_data['Name'].iloc[index], df_comp_params=df_comp_params))
            i_ = i_ + 1

        # create panda tables and array them
        tables_comp = [str] * len(df_comp_records_data)

        inserted_data_df_dummy = df_inserted_data.copy()
        inserted_data_df_dummy['Name'] = 'Inserted spectra'

        i_ = 0
        for _, row in df_comp_records_data.iterrows():
            # the df below (inserted data) is only used to append it to the found matching spectra
            inserted_data_df_dummy['Score [%]'] = row['Score [%]']

            row_df = pd.DataFrame([row])

            row_df['Name'] = '<a href="' + url_for('view_spectra', db_name=db_name, spectra_id=record_ids[i_]) + '" target="_blank">' + row['Name'] + '</a>'

            row_df = pd.concat([row_df, inserted_data_df_dummy], ignore_index=True)

            tables_comp[i_] = row_df.to_html(classes='table table-striped table-bordered table-hover',
                                                     index=False, escape=False)
            i_ = i_ + 1

        return render_template('compare_spectrums.html', file_df=file_df, options=options_selected,
                               comp_set=comp_set, plot_inserted=plot_inserted,
                               table=df_inserted_data.to_html(classes='table table-striped table-bordered table-hover',
                                                     index=False), tables_comp=tables_comp, plots_comp=plots_comp)

    else:
        return render_template('compare_spectrums.html', options=None, comp_set=None, plot_inserted=None)


@app.route('/deleted_spectrum_page', methods=['GET'])
def deleted_spectrum_page():
    db_name = request.args.get('db_name', 0)
    spectra_id = request.args.get('spectra_id', 0)
    context = request.args.get('context', 0) # json struct containing the search params

    if db_name and spectra_id:
        db = DB(db_name)
        if db.delete_record(spectra_id):
            db.close()
            return render_template('deleted_spectrum.html', context=context, successful='Spectra deleted successfully')
        db.close()
        return render_template('deleted_spectrum.html', context=context, error='Error during deletion of the spectrum')

    return render_template('deleted_spectrum.html', context=context, error='No spectrum id or database name inserted')


@app.route('/view_spectra', methods=['GET'])
def view_spectra():
    db_name = request.args.get('db_name', 0)
    spectra_id = request.args.get('spectra_id', 0)
    context = request.args.get('context', 0) # json struct containing the search params
    delete_spectrum = request.args.get('delete_spectrum', 0)
    description = request.args.get('description', 0)

    columns = ['id', 'Spectra name', 'Nr. Peaks', 'Peak wavelengths', 'Peaks FWHM', 'Chosen min_peak_height']

    if delete_spectrum:
        return redirect(url_for('deleted_spectrum_page', db_name=db_name, spectra_id=spectra_id, context=context))

    if db_name and spectra_id:

        if description:
            DB.update_record_description(db_name, spectra_id, description)

        db = DB(db_name)
        df_spectra, record = db.look_up_by_id(spectra_id)
        db.close()
        df_spectra = df_spectra.drop('DB Name', axis=1)
        df_spectra.columns = columns

        wavelengths, intensities = record.get_spectra_high_res()
        df_plot = pd.DataFrame({'Wavelength [nm]': wavelengths, 'Intensity [counts]': intensities})

        locations, width_heights, left_ips, right_ips, fwhm = peaks_spot(wavelengths, intensities,
                                                                         record.min_peak_height)

        plot = create_plot(df_plot, df_spectra['Spectra name'], 2, locations, width_heights, left_ips, right_ips)

        message = (f"Find fluorescent compounds that could be a match with these peak wavelengths = {locations.tolist()} nm and fwhm = {fwhm} nm"
                   f". Hint: the user thinks it is {record.name}. Give me only one straight answer and give me information about the fluorescent compounds")
        print(message)

        spectrum_description = DB.get_record_description(db_name, spectra_id)
        assistant_reply = 0
        if spectrum_description.strip() == "File not found":
            spectrum_description = 0

            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
                model="gpt-4"
            )

            assistant_reply = response.choices[0].message.content
            # assistant_reply = 'intelligenza artificiale'


        return render_template('view_spectra.html', plot=plot,
                               table=df_spectra.to_html(classes='table table-striped table-bordered table-hover',
                                                        index=False),
                               db_name=db_name, spectra_id=spectra_id, context=context, ai_reply=assistant_reply,
                               description=spectrum_description)

    return render_template('view_spectra.html', error='No spectrum id or database name inserted',
                           db_name=db_name, spectra_id=spectra_id, context=context)

@app.route('/search_spectrums', methods=['GET'])
def search_spectrums():
    criteria_list = {'id', 'Name', 'Peak wavelength and FWHM'}
    context = request.args.get('context', 0)

    if context:
        context_unpack = json.loads(context)
        db_name = context_unpack['db_name']
        search_criteria = context_unpack['search_criteria']
        search_criteria_text = context_unpack['search_criteria_text']
        peak_wav_criteria = context_unpack['peak_wav_criteria']
        fwhm_wav_criteria = context_unpack['fwhm_criteria']
        peak_wav_tol = context_unpack['peak_wav_criteria_tol']
        fwhm_tol = context_unpack['fwhm_tol']

        if peak_wav_criteria == '':
            peak_wav_criteria = 0
        else:
            peak_wav_criteria = float(peak_wav_criteria)
        if fwhm_wav_criteria == '':
            fwhm_wav_criteria = 0
        else:
            fwhm_wav_criteria = float(fwhm_wav_criteria)
        if peak_wav_tol == '':
            peak_wav_tol = 0
        else:
            peak_wav_tol = float(peak_wav_tol)
        if fwhm_tol == '':
            fwhm_tol = 0
        else:
            fwhm_tol = float(fwhm_tol)

    else:
        db_name = request.args.get('db_name', 0)
        search_criteria = request.args.get('search_criteria', 0)
        search_criteria_text = request.args.get('search_criteria_text', 0)

        if request.args.get('peak_wav_criteria') == '':
            peak_wav_criteria = 0
        else:
            peak_wav_criteria = float(request.args.get('peak_wav_criteria', 0))
        if request.args.get('fwhm_criteria') == '':
            fwhm_wav_criteria = 0
        else:
            fwhm_wav_criteria = float(request.args.get('fwhm_criteria', 0))
        if request.args.get('peak_wav_criteria_tol') == '':
            peak_wav_tol = 0
        else:
            peak_wav_tol = float(request.args.get('peak_wav_criteria_tol', 0))
        if request.args.get('fwhm_tol') == '':
            fwhm_tol = 0
        else:
            fwhm_tol = float(request.args.get('fwhm_tol', 0))

    context = {'db_name': db_name,
               'search_criteria': search_criteria,
               'search_criteria_text': search_criteria_text,
               'peak_wav_criteria': peak_wav_criteria,
               'fwhm_criteria': fwhm_wav_criteria,
               'peak_wav_criteria_tol': peak_wav_tol,
               'fwhm_tol': fwhm_tol
               }

    id_spectra = 0
    name_spectra = 0
    peak_wav = 0.0
    fwhm = 0.0

    db_list, _ = DB.get_databases_list()

    if not db_name or db_name == 'Choose DB':
        return render_template('search_spectrums.html', db_list=db_list['name'],
                               criteria_list=criteria_list, error='Choose a DB!')

    if search_criteria != 'Peak wavelength and FWHM' and (search_criteria_text == '' or search_criteria_text == 0):
        db = DB(db_name)
        df_spectra = db.get_db()
        df_spectra.columns = ['id', 'Spectra name', 'Nr. Peaks', 'Peak wavelengths', 'Peaks FWHM',
                              'Chosen min_peak_height']

        for index, row in df_spectra.iterrows():
            df_spectra.at[index, 'Spectra name'] = '<a href="' + url_for('view_spectra', db_name=db_name, spectra_id=row['id'], context=json.dumps(context)) + '">' + df_spectra.at[index, 'Spectra name'] + '</a>'

        db.close()
        return render_template('search_spectrums.html', db_list=db_list['name'], db_name=db_name,
                               search_criteria=search_criteria, criteria_list=criteria_list,
                               search_criteria_text=search_criteria_text,
                               table=df_spectra.to_html(classes='table table-striped table-bordered table-hover',
                                                        index=False, escape=False))

    if not search_criteria or search_criteria == 'Choose criteria':
        return render_template('search_spectrums.html', db_list=db_list['name'],
                               criteria_list=criteria_list, error='Choose a search criteria!')

    if search_criteria == 'id':
        id_spectra = int(search_criteria_text)
    elif search_criteria == 'Name':
        name_spectra = str(search_criteria_text)
    elif search_criteria == 'Peak wavelength and FWHM':
        peak_wav = float("{:.2f}".format(peak_wav_criteria))
        fwhm = float("{:.2f}".format(fwhm_wav_criteria))
        peak_wav_tol = float("{:.2f}".format(peak_wav_tol))
        fwhm_tol = float("{:.2f}".format(fwhm_tol))

    if db_name:
        db = DB(db_name)

        columns = ['id', 'Spectra name', 'Nr. Peaks', 'Peak wavelengths', 'Peaks FWHM', 'Chosen min_peak_height']
        if id_spectra:
            df_spectra, fluo_rec_spectra = db.look_up_by_id(id_spectra)
            df_spectra = df_spectra.drop('DB Name', axis=1)
            df_spectra.columns = columns

        elif name_spectra:
            df_spectra = db.look_up_by_name(name_spectra)
            df_spectra.columns = columns
        elif peak_wav and fwhm:
            df_spectra = db.look_up_records_pks_wav_fwhm(peak_wav, fwhm, peak_wav_tol, fwhm_tol)
            df_spectra.columns = columns

        for index, row in df_spectra.iterrows():
            df_spectra.at[index, 'Spectra name'] = '<a href="' + url_for('view_spectra', db_name=db_name, spectra_id=row['id'], context=json.dumps(context)) + '">' + df_spectra.at[index, 'Spectra name'] + '</a>'

        db.close()
        return render_template('search_spectrums.html', db_list=db_list['name'], db_name=db_name,
                               search_criteria=search_criteria, criteria_list=criteria_list,
                               search_criteria_text=search_criteria_text, peak_wav=peak_wav, fwhm=fwhm,
                               peak_wav_tol=peak_wav_tol, fwhm_tol=fwhm_tol,
                               table=df_spectra.to_html(classes='table table-striped table-bordered table-hover',
                                                        index=False, escape=False))

    return render_template('search_spectrums.html', db_list=db_list['name'], criteria_list=criteria_list)


@app.route('/create_db', methods=['GET', 'POST'])
def create_db():
    df_dbs, num_rec_perdb = DB.get_databases_list()

    db_name_links = []
    for _, row in df_dbs.iterrows():
        db_name_links.append('<a href="' + url_for('search_spectrums', db_name=row['name']) + '" target="_blank">' + row['name'] + '</a>')


    df_dbs_table = pd.DataFrame({'Database name': db_name_links, 'Nr. of records': num_rec_perdb})
    # to correct annoying visual effect in which nr. of records is shown as float value
    df_dbs_table['Nr. of records'] = df_dbs_table['Nr. of records'].astype('int')

    if request.method == 'GET':
        successful = request.args.get('successful')
        return render_template('create_db.html',
                               table=df_dbs_table.to_html(classes='table table-striped table-bordered table-hover',
                                                    index=False, escape=False), successful=successful)
    if request.method == 'POST':
        db_name = str(request.form['database_name'])
        if DB.create_db(db_name) == 1:
            return redirect(url_for('create_db', successful='1'))
        else:
            return render_template('create_db.html', error=f"Database with name {db_name} already exists",
                                   table=df_dbs_table.to_html(classes='table table-striped table-bordered table-hover', index=False))
    return render_template('create_db.html', table=df_dbs_table.to_html(classes='table table-striped table-bordered table-hover', index=False))


@app.route('/add_spectra', methods=['GET', 'POST'])
def add_spectra_page():  # put application's code here
    if request.method == 'POST':
        # Handle form submission
        wavs_to_delete = [float(request.form['led_range_b']), float(request.form['led_range_e'])]
        smoothing_wind = int(request.form['smoothing_wind'])
        min_peak_height = float(request.form['min_peak_height'])
        alpha = int(request.form['alpha'])

        dbs_available_df, _ = DB.get_databases_list()
        dbs_available = dbs_available_df['name'].tolist()

        if 'change_spectrum' in request.form:
            change_spectrum = request.form['change_spectrum']
        else:
            change_spectrum = '0'

        # means spectrum change button has been touched, reload the page without spectrum
        if change_spectrum == '1':
            options_selected = {
                'wavs_to_delete': wavs_to_delete,
                'smoothing_wind': smoothing_wind,
                'min_peak_height': min_peak_height,
                'alpha': alpha
            }
            return render_template('add_spectra.html', options=options_selected, plot_original=None,
                                   plot_elaborated=None)

        # 'spectrum' is a request.files, 'file_df' is a hidden request.form
        if 'spectrum' not in request.files and 'file_df' not in request.form:
            return render_template('add_spectra.html', error='No file part', options=None)
        # maybe it has been sent in the post request...
        elif 'file_df' in request.form:
            df = pd.read_json(StringIO(request.form['file_df']), orient='split')
            df.columns = ['Wavelength [nm]', "Intensity [counts]", "not interested1", "not interested2"]
            filename = request.form['filename']
        # must have been inserted now, read it and check if valid
        else:
            file = request.files['spectrum']
            # Check if the file is not empty
            if file.filename == '':
                return render_template('add_spectra.html', error='No selected file', options=None)
            filename = file.filename

            # Check if the file is allowed (optional)
            allowed_extensions = {'csv', 'txt'}
            if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                return render_template('add_spectra.html', error='Invalid file extension', options=None)

            # Read the file using Pandas
            try:
                if file.filename.endswith('.csv') or file.filename.endswith('.txt'):
                    df = pd.read_csv(file, sep=";", header=5)
                    df.columns = ['Wavelength [nm]', "Intensity [counts]", "not interested1", "not interested2"]
                else:
                    return render_template('add_spectra.html', error='Unsupported file format', options=None)
            except pd.errors.EmptyDataError:
                return render_template('add_spectra.html', error='File is empty', options=None)

        # Create an interactive plot using Plotly
        plot_original = create_plot(df, 'Original spectrum', 1)

        # Process data
        add_spectra_class = AddSpectra(wavs_to_delete, smoothing_wind, min_peak_height, alpha, df)
        wavelengths_reduced, intensities_reduced, wavelengths, intensities, locations, fwhm, width_heights, left_ips, right_ips = add_spectra_class.analyze_spectra()

        df_reduced = pd.DataFrame({'Wavelength [nm]': wavelengths_reduced, 'Intensity [counts]': intensities_reduced})
        df_data = pd.DataFrame({'Peak wavelengths [nm]': locations, 'FWHM peaks [nm]': fwhm})

        plot_elaborated = create_plot(df_reduced, 'Conditioned spectrum', 2, locations, width_heights, left_ips, right_ips)

        options_selected = {
            'wavs_to_delete': wavs_to_delete,
            'smoothing_wind': smoothing_wind,
            'min_peak_height': min_peak_height,
            'alpha': alpha,
            'file': filename,
            'dbs': dbs_available
        }

        file_df = df.to_json(orient='split')

        if 'save_spectrum' in request.form and 'db_select' in request.form:
            save_spectrum = request.form['save_spectrum']
            db_selected = request.form['db_select']

            if save_spectrum == '1':
                spectra_name = request.form['save_spectrum_name']
                if spectra_name != '':
                    id_ = add_spectra_class.save_spectra(db_selected, spectra_name, wavelengths, intensities, locations, fwhm,
                                                         min_peak_height)
                    if id_:
                        return redirect(url_for('result_add_successful', name=spectra_name, name_spectra_db=db_selected, id_=id_))
                    else:
                        return render_template('add_spectra.html', file_df=file_df, options=options_selected,
                                               plot_original=plot_original, plot_elaborated=plot_elaborated,
                                               table=df_data.to_html(
                                                   classes='table table-striped table-bordered table-hover',
                                                   index=False),
                                               error='There is some problem with the file you inserted')
                else:
                    return render_template('add_spectra.html', file_df=file_df, options=options_selected,
                                           plot_original=plot_original, plot_elaborated=plot_elaborated,
                                           table=df_data.to_html(
                                               classes='table table-striped table-bordered table-hover',
                                               index=False),
                                           error='You did not insert a name for the spectrum!')

        return render_template('add_spectra.html', file_df=file_df, options=options_selected,
                               plot_original=plot_original, plot_elaborated=plot_elaborated,
                               table=df_data.to_html(classes='table table-striped table-bordered table-hover',
                                                     index=False))

    else:
        return render_template('add_spectra.html', options=None, plot_original=None, plot_elaborated=None)


@app.route('/result_add_successful')
def result_add_successful():
    name_spectra = request.args.get('name')
    name_spectra_db = request.args.get('name_spectra_db')
    id_ins_spectra = request.args.get('id_')
    db = DB(name_spectra_db)
    df_spectra, fluo_rec_spectra = db.look_up_by_id(id_ins_spectra)
    db.close()

    spectra_data = fluo_rec_spectra.get_spectra_high_res()
    locations, width_heights, left_ips, right_ips, fwhm = peaks_spot(spectra_data[0], spectra_data[1],
                                                                     fluo_rec_spectra.min_peak_height)

    df_plot = pd.DataFrame({'Wavelength [nm]': spectra_data[0], 'Intensity [counts]': spectra_data[1]})
    plot_elaborated = create_plot(df_plot, 'Added spectrum', 2, locations, width_heights, left_ips, right_ips)

    df_spectra.columns = ['DB name', 'id', 'Name', 'Number of peaks', 'Wavelengths of Peaks [nm]', 'FWHMs of Peaks [nm]', 'Used min_peak_height']

    return render_template('result_add.html', name_spectra=name_spectra, plot=plot_elaborated,
                           table=df_spectra.to_html(classes='table table-striped table-bordered table-hover', index=False))


def create_plot(df, plot_type, locations=None, width_heights=None, left_ips=None, right_ips=None, df_comp=None, df_comp_name=None, df_comp_params=None):
    # Example: Create a scatter plot using Plotly Express
    if width_heights is None:
        width_heights = []
    fig = px.line(df, x='Wavelength [nm]', y='Intensity [counts]')

    # plot type 1: simple spectrum without annotations (peaks, fwhm)
    # plot type 2: with annotations
    # plot type 3: overlap of two spectras
    if plot_type == 1:
        fig.update_layout(showlegend=True)
    if plot_type == 2 or plot_type == 3:
        fig.add_trace(go.Scatter(x=locations, y=df['Intensity [counts]'][df[df['Wavelength [nm]'].isin(locations)].index], mode='markers', marker=dict(color='red'), showlegend=False))
        for i, _ in enumerate(locations):
            fig.add_shape(
                type='line',
                x0=left_ips[i],
                x1=right_ips[i],
                y0=width_heights[i],
                y1=width_heights[i],
                line=dict(color='green', width=2)
            )
            fig.update_layout(title='Plot with Peaks and FWHM',
                              xaxis_title='Wavelength [nm]',
                              yaxis_title='Intensity [normalized]', showlegend=True)

    if plot_type == 3:
        fig.add_scatter(x=df_comp['Wavelength [nm]'], y=df_comp['Intensity [counts]'], mode='lines', name=df_comp_name)
        fig.add_trace(
            go.Scatter(x=df_comp_params['locations'], y=df_comp['Intensity [counts]'][df_comp['Wavelength [nm]'].isin(df_comp_params['locations'])],
                       mode='markers', marker=dict(color='red'), showlegend=False))
        for i, _ in enumerate(df_comp_params['locations']):
            fig.add_shape(
                type='line',
                x0=df_comp_params['left_ips'][i],
                x1=df_comp_params['right_ips'][i],
                y0=df_comp_params['width_heights'][i],
                y1=df_comp_params['width_heights'][i],
                line=dict(color='green', width=2)
            )
            fig.update_layout(title='Plot with Peaks and FWHM',
                              xaxis_title='Wavelength [nm]',
                              yaxis_title='Intensity [normalized]', showlegend=True)

    # Convert the plot to HTML for rendering in the template
    plot_html = fig.to_html(full_html=False)

    return plot_html


if __name__ == '__main__':
    app.run()
