import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:

    def __init__(self, data, item_features, weighting=True):

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].sum().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].sum().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        # Топ покупок по всему датасету >7$
        self.overall_top_purchases_exp = self.top_all_purchases_exp(data, price=7)

        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        # Словарь {item_id: 0/1}. 0/1 - факт принадлежности товара к СТМ
        self.item_id_to_ctm = dict(zip(item_features['item_id'], item_features['brand'].isin(['Private']).astype(int)))

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T, K1=4, B=0.15).T # , K1=4, B=0.15

        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

        self.model = self.fit(self.user_item_matrix)

        self.item_factors = self.model.item_factors
        self.user_factors = self.model.user_factors

        self.items_emb_df, self.users_emb_df = self.get_embeddings(self)


    @staticmethod
    def prepare_matrix(data, val='quantity'):

        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values=val,  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def get_embeddings(self):
        items_embeddings = pd.DataFrame(self.item_factors)
        items_embeddings['item_id'] = items_embeddings.apply(lambda x: self.id_to_itemid[x.name], axis=1)

        users_embeddings = pd.DataFrame(self.user_factors)
        users_embeddings['user_id'] = users_embeddings.apply(lambda x: self.id_to_userid[x.name], axis=1)

        return items_embeddings, users_embeddings

    @staticmethod
    def top_all_purchases_exp(data, price=7):
        overall_top_purchases_exp = data[data['price'] > price].groupby('item_id')['sales_value'].sum().reset_index()
        overall_top_purchases_exp.sort_values('sales_value', ascending=False, inplace=True)
        overall_top_purchases_exp = overall_top_purchases_exp[overall_top_purchases_exp['item_id'] != 999999]
        overall_top_purchases_exp = overall_top_purchases_exp.item_id.tolist()

        return overall_top_purchases_exp

    @staticmethod
    def top_purchases_exp(data, user_id, price=7):
        top_purchases_exp = \
        data[(data['price'] > price) & (data['user_id'] == user_id)].groupby(['user_id', 'item_id'])[
            'quantity'].count().reset_index()
        top_purchases_exp.sort_values('quantity', ascending=False, inplace=True)
        top_purchases_exp = top_purchases_exp[top_purchases_exp['item_id'] != 999999]
        top_purchases_exp = top_purchases_exp.item_id.tolist()

        return top_purchases_exp

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=0)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.1, iterations=150, num_threads=0):
        """Обучает ALS"""

        # AlternatingLeastSquares
        # BayesianPersonalizedRanking
        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)

        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():

            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _extend_with_top_popular_exp(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases_exp[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        #self._update_dict(user_id=user)
        if user in self.userid_to_id.keys():
            res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                                                        user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                                                        N=N,
                                                                        filter_already_liked_items=False,
                                                                        filter_items=[self.itemid_to_id[999999]],
                                                                        recalculate_user=True)]
        else:
            res = []

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        #self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        #self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_top_purchases(self, data, user, price=7, N=5):
        if user in self.userid_to_id.keys():
            res = self.top_purchases_exp(data=data, user_id=user, price=price)[:N]
        else:
            res = []

        res = self._extend_with_top_popular_exp(res, N=N)

        return res

    def get_similar_items_recommendation(self, user, filter_ctm=True, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        rec_model = self.model.similar_items(self.itemid_to_id[user], N=N)
        recs = [x[0] for x in rec_model][1:]  # получаем список рекомендаций
        rec_to_itemid = [self.id_to_itemid[x] for x in recs]  # переводим в изначальные id

        if filter_ctm:
            ctm_list = [self.item_id_to_ctm[x] for x in rec_to_itemid]  # Список является или нет товар СТМ

            try:
                idx = ctm_list.index(1)  # Берем первый товар СТМ
            except ValueError:
                idx = 0  # либо просто первый, если СТМ в списке не оказалось
        else:
            idx = 0

        res = rec_to_itemid[idx]

        return res

    def get_similar_users_recommendation(self, user, N_users=5, N_items_per_user=5, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        rec_model_users = self.model.similar_users(self.userid_to_id[user], N=N_users+1)
        recs_users = [x[0] for x in rec_model_users][1:]  # получаем список Юзеров

        total_recs = []
        for similar_user in recs_users:
            own = ItemItemRecommender(K=1, num_threads=0)  # K - кол-во билжайших соседей
            own.fit(csr_matrix(self.user_item_matrix).T.tocsr())

            recs = own.recommend(userid=self.userid_to_id[similar_user],  # Находим купленые ими товары
                                     user_items=csr_matrix(self.user_item_matrix).tocsr(),  # на вход user-item matrix
                                     N=N_items_per_user,
                                     filter_already_liked_items=False,
                                     filter_items=None,
                                     recalculate_user=False)
            total_recs.append(recs)

        total_recs = [item for sublist in total_recs for item in sublist]  # делаем общий список товаров
        total_recs = sorted(total_recs, key=lambda l: l[1], reverse=True)  # выбираем товары с большим мкором

        res = [self.id_to_itemid[x[0]] for x in total_recs][:N]  # берем нужное кол-во и переводим в изначальные id

        return res