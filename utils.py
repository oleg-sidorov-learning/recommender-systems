import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# Предфильтрация товаров
def prefilter_items(data, item_features, take_n_popular=5000, min_price=1):

    # 0. Удаляем позиции у которых количество = 0
    idx = data.loc[data['quantity'] == 0].index
    data.drop(idx, inplace=True)
    data.reset_index(drop=True, inplace=True)

    df_grouped = data.groupby(by=['item_id'], as_index=False).sum()
    df_grouped['price'] = df_grouped['sales_value'] / df_grouped['quantity']

    data['price'] = data[['item_id']].merge(df_grouped[['item_id', 'price']], how='left', on='item_id')['price']

    # 1.2. Удаление товаров, со средней ценой < 1$ and > 25$
    data = data[data['price'] > min_price]
    data = data[data['price'] < 15]
    data.reset_index(drop=True, inplace=True)

    # 3. Придумайте свой фильтр (убраны товары, которые не продавались 1 год)
    df_grouped = data.groupby(by=['item_id']).max().sort_values('day')
    idx_sold_12m = data[data['item_id'].isin(df_grouped.loc[df_grouped['day'] > (df_grouped['day'].max() - 30)].index)].item_id.tolist()
    data = data[data['item_id'].isin(idx_sold_12m)]
    data.reset_index(drop=True, inplace=True)

    # 4. Выбор топ-N самых популярных товаров (N = take_n_popular)
    df_grouped = data.groupby(by=['item_id']).nunique()[['user_id']].sort_values('user_id', ascending=False)
    df_grouped.rename(columns={"user_id": "uniq_users"})
    # = data.groupby(by=['item_id']).sum().sort_values('quantity', ascending=False)
    idx_top_k = data[data['item_id'].isin(df_grouped.head(take_n_popular).index)].item_id.tolist()

    data.loc[~data['item_id'].isin(idx_top_k), 'item_id'] = 999999
    data.reset_index(drop=True, inplace=True)

    return data


# Постобработка списка рекомендаций
def postfilter_items(user_id, data, t_price, recommender, item_id_to_price, item_id_to_sub_commodity, user_purchased_items):
    data = np.array(data)
    if np.isnan(data).any():
        data = np.array([])
    data = np.concatenate((data, np.array(recommender.overall_top_purchases)), axis=None)
    if user_id in user_purchased_items.keys():
        user_purchased_items = user_purchased_items[user_id]
    else:
        user_purchased_items = []
    more_then_7 = 0
    not_purch_2 = 0
    comm = []
    postfilter_list = []

    for x in data:
        if item_id_to_price[x] > 7:
            comm.append(item_id_to_sub_commodity[x])
            postfilter_list.append(x)
            more_then_7 += 1
            if x not in user_purchased_items:
                not_purch_2 += 1
            break

    for x in data:
        if (x not in user_purchased_items) and (x not in postfilter_list) and (item_id_to_sub_commodity[x] not in comm):
            comm.append(item_id_to_sub_commodity[x])
            postfilter_list.append(x)
            not_purch_2 += 1
        if not_purch_2 == 2:
            break

    # t_price - ограничение минимальной стоимости товара
    for x in data:
        if (x not in postfilter_list) and (item_id_to_sub_commodity[x] not in comm) and (item_id_to_price[x] > t_price):
            comm.append(item_id_to_sub_commodity[x])
            postfilter_list.append(x)
        if len(postfilter_list) == 5:
            break

    assert len(pd.unique(comm)) == 5, "Not all from diff departments"
    assert not_purch_2 >= 2, "No 2 new items"
    assert more_then_7 >= 1, "No >7$ items"
    assert len(postfilter_list) >= 5, "Less then 5 recommendation"
    return postfilter_list


# Словарь: {item_id: цена данного товара}
def get_price_list(data):
    grouped = data.groupby(by='item_id')['price'].mean().reset_index()
    price_dict = dict(zip(grouped['item_id'], grouped['price']))

    return price_dict


# Словарь: {item_id: параметр sub_commodity_desc для данного товара}
def get_sub_commodity_desc_list(data, item_features):
    grouped = data.groupby(by='item_id')['price'].mean().reset_index()
    grouped = grouped.merge(item_features[['item_id', 'sub_commodity_desc']], how='left', on='item_id')
    commodity_dict = dict(zip(grouped['item_id'], grouped['sub_commodity_desc']))

    return commodity_dict


# Словарь: {user_id: прошлые покупки юзера}
def user_purchased_items_list(data):
    grouped = data.groupby(by='user_id')['item_id'].unique().reset_index()
    purchased_dict = dict(zip(grouped['user_id'], grouped['item_id']))

    return purchased_dict


# Отображение важности фичей у LightGBM
def show_feature_importances(feature_names, feature_importances, get_top=None):
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    feature_importances = feature_importances.sort_values('importance', ascending=False)

    plt.figure(figsize=(20, len(feature_importances) * 0.355))

    sns.barplot(feature_importances['importance'], feature_importances['feature'])

    plt.xlabel('Importance')
    plt.title('Importance of features')
    plt.show()

    if get_top is not None:
        return feature_importances['feature'][:get_top].tolist()