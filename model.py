import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")

import pandas as pd
import numpy as np
import datetime as dt
from collections import defaultdict, deque
import copy

import hyperopt
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe

import lightgbm as lgb
from sklearn.model_selection import train_test_split


def run(info, train_data, train_label, test_data, time_remain):
    # Begin


    # In[ ]:





    # In[7]:


    MAIN_TABLE_NAME = "main"

    TIME_PREFIX = "t_"
    NUMERICAL_PREFIX = "n_"
    CATEGORY_PREFIX = "c_"
    MULTI_CAT_PREFIX = "m_"


    # In[ ]:





    # In[8]:


    def print_one_table_info(table_name, table):
        columns = table.columns
        time_num = sum([1 for col in columns if col.startswith(TIME_PREFIX)])
        numerical_num = sum([1 for col in columns if col.startswith(NUMERICAL_PREFIX)])
        category_num = sum([1 for col in columns if col.startswith(CATEGORY_PREFIX)])
        multi_cat_num = sum([1 for col in columns if col.startswith(MULTI_CAT_PREFIX)])
        label_num = sum([1 for col in columns if col.startswith("label")])
        print(table_name, table.shape,
              "TIME", time_num,
              "NUMERICAL", numerical_num,
              "CATEGORY", category_num,
              "MULTI_CAT", multi_cat_num,
              "LABEL", label_num)
        return

    def print_table_info(tables):
        for table_name in tables:
            print_one_table_info(table_name, tables[table_name])
        return


    # In[ ]:





    # In[9]:


    print_table_info(train_data)
    print("train_label", len(train_label))
    print("test_data", test_data.shape)


    # In[ ]:





    # In[10]:


    for relation in info['relations']:
        print(relation)


    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[11]:


    config = info
    all_data = train_data
    X_train = all_data[MAIN_TABLE_NAME]
    X_train.sort_values(by=config['time_col'], ascending=True, inplace=True)
    X_train['label'] = train_label
    X_train['label_order'] = [i for i in range(len(X_train))]
    X_test = test_data
    X_test['label'] = -1
    X_test['label_order'] = [i for i in range(len(X_test))]
    main_table = pd.concat([X_train, X_test], sort=True)
    all_data[MAIN_TABLE_NAME] = main_table


    # In[ ]:





    # In[12]:


    print_table_info(all_data)


    # In[ ]:





    # In[13]:


    def get_relation_keys(info):
        relation_keys = []
        for d in info['relations']:
            relation_keys += d['key']
        return set(relation_keys)

    def clean_tables(tables):
        relation_keys = get_relation_keys(info)
        for table_name in tables:
            clean_df(tables[table_name], relation_keys)
        return

    def clean_df(df, relation_keys):
        fillna(df)
        clean_memory(df, relation_keys)
        return

    def fillna(df):
        columns = df.columns.tolist()
        none_time = dt.datetime(1970, 1, 1)
        for col in columns:
            if col.startswith(NUMERICAL_PREFIX):
                df[col].fillna(-1.0, inplace=True)
            if col.startswith(CATEGORY_PREFIX):
                df[col].fillna("None", inplace=True)
            if col.startswith(TIME_PREFIX):
                df[col].fillna(none_time, inplace=True)
            if col.startswith(MULTI_CAT_PREFIX):
                df[col].fillna("None", inplace=True)
        return

    def new_category(df, c):
        items = df[c].tolist()
        item_dict = {}
        total_num = 0
        for item in items:
            if item not in item_dict:
                total_num += 1
                item_dict[item] = total_num
        df[c] = df[c].map(item_dict)
        return

    def clean_memory(df, relation_keys):
        df.info(memory_usage='deep')
        columns = df.columns.tolist()
        for col in columns:
            if col.startswith(CATEGORY_PREFIX) and col not in relation_keys:
                new_category(df, col)
        df.info(memory_usage='deep')
        print('-' * 80)
        return


    # In[ ]:





    # In[14]:


    clean_tables(all_data)


    # In[ ]:





    # In[15]:


    def extract_tables(tables):
        for table_name in tables:
            extract_df(tables[table_name])
        return

    def extract_df(df):
        extract_multi_cat_col(df)
        return

    def extract_multi_cat_col(df):
        columns = df.columns.tolist()
        for col in columns:
            if col.startswith(MULTI_CAT_PREFIX):
                extract_col(df, col)
        return

    def extract_col(df, col):
        top_extract_num = 10
        origin_features = df[col].tolist()
        feature_count = defaultdict(int)
        for f in origin_features:
            numbers = f.split(',')
            for n in numbers:
                feature_count[n] += 1
        items = list(feature_count.items())
        items.sort(key=lambda x: (x[1], x[0]), reverse=True)
        items = items[:top_extract_num]
        feature_dict = dict(items)
        new_feature_col = defaultdict(list)
        for f in origin_features:
            numbers = set(f.split(','))
            for num in feature_dict:
                feature = 0
                if num in numbers:
                    feature = 1
                new_feature_col[num].append(feature)
        df.drop(col, axis=1, inplace=True)
        for num in feature_dict:
            df["n_" + col + "_" + num] = new_feature_col[num]
        return


    # In[ ]:





    # In[16]:


    print_table_info(all_data)


    # In[ ]:





    # In[17]:


    extract_tables(all_data)


    # In[ ]:





    # In[18]:


    print_table_info(all_data)


    # In[ ]:





    # In[19]:


    def bfs(graph, tconfig):
        tconfig[MAIN_TABLE_NAME]['depth'] = 0
        queue = deque([MAIN_TABLE_NAME])
        while queue:
            u_name = queue.popleft()
            for edge in graph[u_name]:
                v_name = edge['to']
                if 'depth' not in tconfig[v_name]:
                    tconfig[v_name]['depth'] = tconfig[u_name]['depth'] + 1
                    queue.append(v_name)
        return

    def aggregate_op(col):
        ops = {
            TIME_PREFIX: ["count"],
            NUMERICAL_PREFIX: ["mean"],
            CATEGORY_PREFIX: ["count"]
        }
        if col.startswith(TIME_PREFIX):
            return ops[TIME_PREFIX]
        if col.startswith(NUMERICAL_PREFIX):
            return ops[NUMERICAL_PREFIX]
        if col.startswith(CATEGORY_PREFIX):
            return ops[CATEGORY_PREFIX]
        return None

    def dfs(u_name, graph, config, tables):
        u = tables[u_name]
        for edge in graph[u_name]:
            v_name = edge['to']
            if config['tables'][v_name]['depth'] <= config['tables'][u_name]['depth']:
                continue

            v = dfs(v_name, graph, config, tables)
            key = edge['key']
            type_ = edge['type']

            if type_.split("_")[2] == 'one':
                v = v.set_index(key)
                v.columns = v.columns.map(lambda a: "{0}_{1}_{2}".format(a.split('_', 1)[0], v_name, a))
                u = u.join(v, on=key, how='left')
                del v
            else:
                agg_funcs = {col: aggregate_op(col) for col in v.columns if col != key}
                v = v.groupby(key).agg(agg_funcs)
                v.columns = v.columns.map(lambda a: "{0}{1}_{2}_{3}".format(NUMERICAL_PREFIX, v_name, a[0], a[1]))
                u = u.join(v, on=key, how='left')
                del v
        return u

    def merge_table(tables, config):
        graph = defaultdict(list)
        for rel in config['relations']:
            ta = rel['table_A']
            tb = rel['table_B']
            graph[ta].append({
                "to": tb,
                "key": rel['key'],
                "type": rel['type']
            })
            graph[tb].append({
                "to": ta,
                "key": rel['key'],
                "type": '_'.join(rel['type'].split('_')[::-1])
            })
        bfs(graph, config['tables'])
        df = dfs(MAIN_TABLE_NAME, graph, config, tables)
        return df


    # In[ ]:





    # In[20]:


    print_table_info(all_data)


    # In[ ]:





    # In[21]:


    X = merge_table(all_data, config)


    # In[ ]:





    # In[22]:


    print_one_table_info("X", X)


    # In[23]:


    X.info(memory_usage='deep')


    # In[ ]:





    # In[24]:


    del all_data


    # In[ ]:





    # In[25]:


    def new_category_without_small(train_df, df, c):
        small_num = 50
        items = train_df[c].tolist()
        item_dict = {}
        item_count = defaultdict(int)
        total_num = 0
        for item in items:
            item_count[item] += 1
            if item not in item_dict:
                total_num += 1
                item_dict[item] = total_num
        for item in item_count:
            if item_count[item] < small_num:
                item_dict[item] = 0
        items = df[c].tolist()
        for item in items:
            if item not in item_dict:
                item_dict[item] = 0
        df[c] = df[c].map(item_dict)
        return

    def transform_categorical_hash(df):
        train_df = df[df['label'] >= 0]
        for c in df.columns:
            if c.startswith(CATEGORY_PREFIX):
                new_category_without_small(train_df, df, c)
        return

    def transform_datetime(df, time_col):
        for c in df.columns:
            if c.startswith(TIME_PREFIX) and c != time_col:
                df["n_" + c] = df[c].apply(lambda x: x.timestamp())
                df.drop(c, axis=1, inplace=True)
            if c == time_col:
                df["n_" + c + "_hour"] = df[c].apply(lambda x: x.hour)
                df["n_" + c + "_weekday"] = df[c].apply(lambda x: x.weekday())
                df.drop(c, axis=1, inplace=True)
        return

    def feature_engineer(df, config):
        time_col = config['time_col']
        transform_categorical_hash(df)
        transform_datetime(df, time_col)
        return


    # In[ ]:





    # In[26]:


    print(X.dtypes.value_counts())


    # In[27]:


    print_one_table_info("X", X)


    # In[ ]:





    # In[28]:


    feature_engineer(X, config)


    # In[ ]:





    # In[29]:


    print(X.dtypes.value_counts())


    # In[30]:


    print_one_table_info("X", X)


    # In[31]:


    print(X.shape)


    # In[ ]:





    # In[32]:


    X_train = X[X['label'] >= 0].copy()
    X_train.sort_values(by='label_order', ascending=True, inplace=True)
    X_test = X[X['label'] == -1].copy()
    X_test.sort_values(by='label_order', ascending=True, inplace=True)
    del X


    # In[ ]:





    # In[33]:


    print(X_train.shape)
    print(X_test.shape)


    # In[34]:


    X_train.info(memory_usage='deep')
    X_test.info(memory_usage='deep')


    # In[ ]:





    # In[35]:


    def print_unknown_feature(X_train, X_test):
        columns = X_train.columns.tolist()
        for col in columns:
            if col.startswith(CATEGORY_PREFIX):
                train_features = set(X_train[col].tolist())
                test_features = set(X_test[col].tolist())
                unknown_features = test_features - train_features
                print(col, len(unknown_features), len(train_features), len(test_features))
        return


    # In[ ]:





    # In[36]:


    print_unknown_feature(X_train, X_test)


    # In[ ]:





    # In[37]:


    def train(X):
        return train_lightgbm(X)

    def predict(X, model, features):
        preds = model.predict(X[features])
        return preds

    def train_lightgbm(X):
        features = [col for col in X.columns if not col.startswith('label')]
        categorical_feature = [col for col in X.columns.tolist() if col.startswith(CATEGORY_PREFIX)]
        X_sample = X[features]
        y_sample = X['label']

        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4
        }
        hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, categorical_feature)

        X_train, X_val, y_train, y_val = data_split(X_sample, y_sample, 0.25, True)
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_feature, free_raw_data=False)
        valid_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_feature, free_raw_data=False)

        model = lgb.train({**params, **hyperparams},
                            train_data,
                            500,
                            valid_data,
                            early_stopping_rounds=30,
                            verbose_eval=100)
        return model, features

    def hyperopt_lightgbm(X, y, params, categorical_feature):
        X_train, X_val, y_train, y_val = data_split(X, y, 0.5, False)
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_feature, free_raw_data=False)
        valid_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_feature, free_raw_data=False)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
            "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
            "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
            "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 300,
                              valid_data, early_stopping_rounds=30, verbose_eval=0)
            score = model.best_score["valid_0"][params["metric"]]
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=tpe.suggest, max_evals=10, verbose=1,
                             rstate=np.random.RandomState(1))

        hyperparams = space_eval(space, best)
        print("auc = {0:0.4f} {1}".format(-trials.best_trial['result']['loss'], hyperparams))
        return hyperparams

    def data_split(X, y, test_size, shuffle):
        return train_test_split(X, y, test_size=test_size, random_state=1, shuffle=shuffle)


    # In[ ]:





    # In[38]:


    model, features = train(X_train)


    # In[ ]:





    # In[39]:


    del X_train


    # In[ ]:





    # In[40]:


    scores = predict(X_test, model, features)
    result = pd.Series(scores)
    print("result: ", len(result))


    # In[ ]:





    # In[41]:


    del X_test


    # In[ ]:





    # In[42]:


    print(result.head(20))


    # In[ ]:





    # In[43]:


    # End
    return result


class Model:
    def __init__(self, info):
        self.info = info
        return

    def fit(self, train_data, train_label, time_remain):
        self.train_data = train_data
        self.train_label = train_label
        return

    def predict(self, test_data, time_remain):
        return run(self.info, self.train_data, self.train_label, test_data, time_remain)
