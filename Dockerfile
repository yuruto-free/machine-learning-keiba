FROM python:3.9.7-buster
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Asia/Tokyo

# Install packages and setup timezone
RUN    apt-get update \
    && apt-get install -y tzdata \
    && ln -sf /usr/share/zoneinfo/${TZ} /etc/localtime \
    && useradd -m labuser \
    && mkdir -p /home/labuser/work \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN    apt-get update \
    && apt-get install -y cargo \
    \
    # install jupyter packages
    \
    && pip install --no-cache-dir \
        black \
        jupyterlab \
        jupyterlab_code_formatter \
        jupyterlab-git \
        lckr-jupyterlab-variableinspector \
        jupyterlab_widgets \
        ipywidgets \
        import-ipynb \
    \
    # install basic packages
    \
    && pip install --no-cache-dir \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        pycaret \
        matplotlib \
        japanize_matplotlib \
        mlxtend \
        seaborn \
        plotly \
        requests \
        beautifulsoup4 \
        lxml \
        html5lib \
        Pillow \
        opencv-python \
    \
    # install additional packages
    \
    && pip install --no-cache-dir \
        pydeps \
        graphviz \
        pandas_profiling \
        shap \
        umap \
        xgboost \
        optuna \
        MonthDelta \
        lightgbm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--notebook-dir=/home/labuser/work"]
