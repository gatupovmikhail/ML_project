git clone https://github.com/gatupovmikhail/ML_project && \
cd ML_project && \
conda env create --file gatupov.yml python=3.7.10 && \
conda activate gatupov && \
chmod u+x down_test.sh && \
bash down_test.sh && \
pytnon3 app.py
