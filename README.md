houseprice
==============================

A short description of the project.

Project Organization
------------

    ├── README.md               <- The top-level README for developers using this project.
    │
    ├── docs                    <- A place to store documentation for the project
    │
    ├── notebooks               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                              the creator's initials, and a short `-` delimited description, e.g.
    │                              `1.0-jqp-initial-data-exploration`.
    │
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures             <- Generated graphics and figures to be used in reporting
    │
    ├── modelling               <- Source code to develop the models and generate reports
    │   ├── requirements.txt    <- Hold requirements for the project
    │   ├── __init__.py         <- Makes modelling a Python module
    │   │
    │   ├── data.py             <- Objects to download or generate data
    │   │
    │   ├── models.py           <- Objects to define, train models and then use trained models to make
    │   │                          predictions
    │   │
    │   ├── visualize.py        <- Objects to create exploratory and results oriented visualizations
    │   │
    │   └── app.py              <- Application to execute everything from the command line
    │
    ├── service                 <- Source code to expose your model as a service
    │   ├── requirements.txt    <- Makes src a Python module
    │   ├── __init__.py         <- Makes service a Python module
    │   │
    │   └── app.py              <- Objects to download or generate data
    │
    ├── tests                   <- Where you check that your code actually works as expected
    │
    └── setup.cfg               <- File holding settings for your project


--------

Example for testing the Fastapi service:

[
  {
    "season": 1,
    "yr": 2012,
    "mnth": 6,
    "hr": 8,
    "holiday": 0,
    "weekday": 1,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0,
    "atemp": 0,
    "hum": 0,
    "windspeed": 0,
    "cnt_1_hours": 20,
    "cnt_2_hours": 25,
    "cnt_3_hours": 30,
    "cnt_7_days": 50,
    "cnt_last_hour_diff": 5,
    "holiday_7_days": 0,
    "workingday_7_days": 1,
    "temp_1_hours": 0.42,
    "temp_2_hours": 0.42,
    "temp_3_hours": 0.42,
    "atemp_1_hours": 0.41,
    "atemp_2_hours": 0.41,
    "atemp_3_hours": 0.41,
    "hum_1_hours": 0.7,
    "hum_2_hours": 0.7,
    "hum_3_hours": 0.7,
    "windspeed_1_hours": 0,
    "windspeed_2_hours": 0,
    "windspeed_3_hours": 0
  }
]


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
