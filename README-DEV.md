Data Quality dashboard
=========================================================


## Useful resources

* [Tips and best practices about data quality monitoring](https://dqo.ai/blog/): Several articles on how to measure data quality metrics and how to create a data quality dashboard.

*  [How to Create a Data Quality Dashboard](https://towardsdatascience.com/data-quality-dashboard-9c60f72b245c): A nice article defining the most common quality metrics, which have to be considered while developing a data quality dashboard.

* [Why you should build a data quality dashboard: benefits and tips](https://www.cloverdx.com/blog/why-you-should-build-a-data-quality-dashboard-benefits-and-tips): A nice article introducing quality metrics and mention several reasons to build a data quality dashboard for monitoring real-time data.

* [Data Quality Dashboard Eindhoven](https://pure.tue.nl/ws/portalfiles/portal/140459730/2019_10_24_ST_Le_D.pdf): A report discussing the implementation of linked data quality dashboard.

## Follow-up Ideas
In this section, we list all ideas, which emerged during the work and can be used for future extensions. 


# TODOs

* [x] integrate mlflow and version control from Sam
* [x] refine the run_all script to run both scripts while enabling the FDs to be stored in API/results/data_fds
* [x] refactoring the app script by separating callbacks and layouts
* [x] solve the mlflow problem which occurs when trying to run different detectors sequentially.
    * Run with UUID ... is already active. To start a new run, first end the current run with mlflow.end_run(). To start a nested run, call start_run with nested=True
* [x] allow users to define the labeling budget of ML-based detectors
* [x] enable FD rules to be displayed without needing to refresh the dashboard
* [x] implement the iterative cleaning module


* [x] Propose another tuple to the user if no error exist in the current tuple
* [x] showing notifications of what is happening in the background, e.g., generating FDs, uploading data, storing file.
* [x] raha causes a problem while generating the strategies

* [x] integrate user-in-the-loop module from Arne
    * [x] enable users to define tags
    * [x] enable users to define rules

* [ ] update README to explain how to integrate new tools and datasets.
    * [ ] how to integrate new tools
    * [x] how to integrate new datasets
* [ ] refactor the code