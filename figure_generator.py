import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances, silhouette_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px


class DfUtil:
    @classmethod
    def export_flatten_df(cls, df):
        """
        导出一个df的平铺格式
        """
        ret = []
        for xlabel, row in df.to_dict().items():
            for ylabel, value in row.items():
                ret.append(dict(xlabel=xlabel, ylabel=ylabel, value=value))
        return ret

    @classmethod
    def choose_strongest_columns(cls, df):
        """
        选取相关性最强的5列
        """
        return abs(df).sum().nlargest(5)

    @classmethod
    def biggest_value_index(cls, df):
        """
        返回df二维表中最大值对应的列标签、行标签
        """
        max_index = np.unravel_index(df.values.argmax(), df.shape)
        return df.columns[max_index[1]], df.index[max_index[0]]

    @classmethod
    def max_value_under_column(cls, series):
        """
        series一列对应的最大值的标签
        """
        return series.idxmax()

    @classmethod
    def get_biggest_chain(cls, df_new, total_n):
        """
        一次查询最大值链
        """
        # 每个标签和自己的相关性置零
        axes = list(cls.biggest_value_index(df_new))
        for i in range(total_n - 2):
            last_ax = axes[-1]

            series = df_new[last_ax]
            for label in axes:
                series[label] = 0

            new_ax = cls.max_value_under_column(series)
            axes.append(new_ax)

        return axes


class RawData:
    raw_data = pd.read_csv("./data/Pokemons.csv", encoding='utf-8')
    number_data = raw_data[["HP", "Att", "Def", "Spd", "Spe", "BST", "Height", "Weight"]]
    dimension = len(number_data.columns)


class KMeansHandler:
    data = np.array(RawData.number_data)
    best_k = None
    labels = None
    silhouette_scores = None

    @staticmethod
    def calculate_best_k(data, max_k=10):
        if KMeansHandler.silhouette_scores is None:
            silhouette_scores = {}
            for k in range(2, max_k):
                kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
                kmeans.fit(data)
                score = silhouette_score(data, kmeans.labels_)
                silhouette_scores[k] = score
            KMeansHandler.silhouette_scores = silhouette_scores
        # 通过silhouette scores找到最佳k值
        best_k = max(KMeansHandler.silhouette_scores, key=KMeansHandler.silhouette_scores.get)
        return best_k

    def get_kmeans_labels(cls):
        # 如果现在没有簇标签就计算一下
        if cls.labels is None or cls.best_k is None:
            cls.best_k = cls.calculate_best_k(cls.data)
            kmeans = KMeans(n_clusters=cls.best_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(cls.data)
            cls.labels = kmeans.labels_
        return cls.best_k, cls.labels


class MDSHandler:
    data = np.array(RawData.number_data)
    mds_points = None

    @classmethod
    def compute_mds_points(cls):
        if cls.mds_points is None:
            mds = MDS(n_components=2, random_state=0, dissimilarity='precomputed')
            cls.mds_points = mds.fit_transform(euclidean_distances(cls.data))
        return cls.mds_points

    @classmethod
    def plot_data_by_euclidian(cls, filename, data_in_range, highlight_color):

        # 对数据集应用MDS
        points = cls.compute_mds_points()
        fig = go.Figure()

        # 根据簇的分类，按不同的颜色画出
        for i in range(best_k):
            cluster_points = points[labels == i]
            fig.add_trace(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                marker=dict(color=colors[i]),
                name=f'Cluster {i}'
            ))

        # Highlight the filtered points
        if data_in_range is not None and highlight_color is not None:
            # 找到 data_in_range 在原始数据集中的索引
            indices_to_highlight = [i for i, point in enumerate(cls.data) if
                                    point.tolist() in data_in_range.values.tolist()]
            # Highlight points
            fig.add_trace(go.Scatter(
                x=points[indices_to_highlight, 0],
                y=points[indices_to_highlight, 1],
                mode='markers',
                marker=dict(color=highlight_color, size=10, symbol='x'),
                name='Highlighted Points'
            ))

        # 更新布局
        fig.update_layout(
            title='MDS with Euclidean distances',
            xaxis_title='MDS 1',
            yaxis_title='MDS 2',
            legend_title='Legend',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        # 如果有filename保存为文件，如果没有显示在界面上
        if filename:
            fig.write_image(filename, scale=2)
        else:
            return fig


class PCAHandler:
    data = np.array(RawData.number_data)
    pca = None
    trans_X = None
    points_2d = None
    variance = None

    @staticmethod
    def my_dot(a_1, a_2):
        return np.dot(
            a_1,
            a_2.T
        ).T

    @classmethod
    def fit_pca_model(cls):
        if cls.pca is None:
            cls.pca = PCA(n_components=RawData.dimension)
            cls.pca.fit(cls.data)
            cls.variance = cls.pca.explained_variance_
        return cls

    @classmethod
    def get_projected_2d_points(cls):
        if cls.points_2d is None:
            cls.fit_pca_model()
            cls.points_2d = cls.my_dot(cls.pca.components_[:2], cls.data - cls.pca.mean_)
        return cls.points_2d

    @classmethod
    def plot_biplot(cls, filename, data_in_range, highlight_color):
        cls.fit_pca_model()
        points = cls.get_projected_2d_points()
        vectors = cls.pca.components_[:2, :]
        x = points[:, 0]
        y = points[:, 1]
        scalex = 1.0 / (x.max() - x.min())
        scaley = 1.0 / (y.max() - y.min())
        feature_names = ["HP", "Att", "Def", "Spd", "Spe", "BST", "Height", "Weight"]
        fig = go.Figure()

        # 对于每个簇将散点画出来
        for label in unique_labels:
            label_points = points[labels == label]
            fig.add_trace(go.Scatter(
                x=label_points[:, 0] * scalex,
                y=label_points[:, 1] * scaley,
                mode='markers',
                marker=dict(color=colors[label % len(colors)]),
                name=f'Cluster {label}'
            ))

        # Highlight 特殊点
        if data_in_range is not None and highlight_color is not None:
            indices_to_highlight = [i for i, point in enumerate(cls.data) if
                                    point.tolist() in data_in_range.values.tolist()]
            fig.add_trace(go.Scatter(
                x=points[indices_to_highlight, 0] * scalex,
                y=points[indices_to_highlight, 1] * scaley,
                mode='markers+text',
                marker=dict(color=highlight_color, size=10, symbol='x'),
                name='Highlighted Points'
            ))

        # 标记特征名字
        for i, vector in enumerate(vectors.T):
            x, y = vector[0], vector[1]
            length = np.sqrt(x ** 2 + y ** 2)  # 计算变量长度
            fig.add_shape(
                type="line",
                x0=0, y0=0,
                x1=x, y1=y,
                line=dict(color='black', width=length),
            )
            fig.add_annotation(
                x=x, y=y,
                text=feature_names[i],
                showarrow=False,
                font=dict(size=12, color='black'),
            )

        # 更新布局
        fig.update_layout(
            title='PCA Biplot',
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            legend_title='Legend',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=0, r=0, b=0, t=40),
            hovermode='closest'
        )

        # 如果有filename保存为文件，如果没有显示在界面上
        if filename:
            fig.write_image(filename, scale=2)
        else:
            return fig


class Correlation:
    cor_matrix = None

    @classmethod
    def get_cor_matrix(cls):
        if cls.cor_matrix is None:
            cls.cor_matrix = RawData.number_data.corr()
        return cls.cor_matrix

    ##################

    @classmethod
    def get_8x8_correlation(cls):
        cor_df = cls.get_cor_matrix()
        return {
            "column_names": list(RawData.number_data.columns),
            "cor_max": DfUtil.export_flatten_df(cor_df)
        }

    @classmethod
    def get_5x5_scatter(cls):
        strong_fields = list(DfUtil.choose_strongest_columns(
            cls.get_cor_matrix()
        ).index)

        return {
            "column_names": strong_fields,
            "values": RawData.number_data[strong_fields].to_dict('records')
        }

    @classmethod
    def get_parallel_axes(cls):
        axes = DfUtil.get_biggest_chain(abs(cls.get_cor_matrix().replace(1, 0)), 8)
        return {
            'column_names': axes,
            'values': RawData.number_data[axes].to_dict('records')
        }

    @classmethod
    def plot_parallel_axes(cls, filename, data_in_range, highlight_color):

        # 定义相关性矩阵和获取坐标轴的方法
        axes = DfUtil.get_biggest_chain(abs(cls.get_cor_matrix().replace(1, 0)), 8)
        data_to_plot = RawData.number_data[axes].copy()

        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_to_plot[axes])
        scaled_data_df = pd.DataFrame(scaled_data, columns=axes)
        data_to_plot.update(scaled_data_df)

        # 假设'labels'是已经定义的簇标签
        data_to_plot['Cluster'] = labels

        # 创建颜色映射
        unique_labels = np.unique(labels)
        label_to_color_scale_value = np.linspace(0, 1, len(unique_labels))
        color_scale = [(value, color) for value, color in zip(label_to_color_scale_value, colors)]

        # 创建一个映射标签到比例尺数值的字典
        label_to_scale_value_map = {label: value for label, value in zip(unique_labels, label_to_color_scale_value)}

        # 为每个数据点指定颜色比例尺上的数值
        line_color_scale_values = [label_to_scale_value_map[label] for label in data_to_plot['Cluster']]
        if data_in_range is not None:
            # 添加一个新的标签，代表 data_in_range 中的数据
            new_label = max(labels) + 1
            data_to_plot.loc[data_in_range.index, 'Cluster'] = new_label
            new_labels = data_to_plot['Cluster']
            new_unique_labels = np.unique(new_labels)
            label_to_color_scale_value = np.linspace(0, 1, len(new_unique_labels))
            colors.insert(new_label, highlight_color)
            color_scale = [ (value, color) for value, color in zip(label_to_color_scale_value, colors)]
            label_to_scale_value_map = {label: value for label, value in zip(new_unique_labels, label_to_color_scale_value)}
            line_color_scale_values = [label_to_scale_value_map[label] for label in data_to_plot['Cluster']]


        fig = go.Figure(data=go.Parcoords())

        # 隐藏坐标轴下方的变量名
        for col in axes:
            fig.update_xaxes(title_text='', showticklabels=False, showgrid=False)

        fig.add_trace(go.Parcoords(
            line=dict(
                color=line_color_scale_values,  # 应用颜色比例尺映射的数值
                colorscale=color_scale,  # 定义颜色比例尺
                showscale=False,  # 不显示颜色比例尺

            ),
            dimensions=[dict(
                range=[scaled_data_df[col].min(), scaled_data_df[col].max()],
                label=col, values=scaled_data_df[col]
            ) for col in axes]
        ))

        # 保存或返回图形
        if filename:
            fig.write_image(filename)
        else:
            return fig


def area_plot():
    # 读取CSV文件
    df = RawData.raw_data

    # 创建面积图
    fig = go.Figure()

    # 添加面积图轨迹
    fig.add_trace(
        go.Scatter(x=df['Weight'].iloc[550:], y=df['Def'], mode='lines', fill='tozeroy', stackgroup='one', name='def'))
    fig.add_trace(
        go.Scatter(x=df['Weight'].iloc[550:], y=df['Spd'], mode='lines', fill='tozeroy', stackgroup='one', name='spd'))

    # 设置图表布局
    fig.update_layout(
        title='Area Graph with Weight, Def, and Spd',
        xaxis=dict(title='Weight'),
        yaxis=dict(title='Values'),
        legend_title='Legend',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


# 保存数据的簇标签
kmeans_handler = KMeansHandler()
best_k, labels = kmeans_handler.get_kmeans_labels()
unique_labels = np.unique(labels)
# 为簇设置的颜色列表
colors = ['#f6b57b', '#2b9fc9', '#FF8FAB', 'Green', 'Black', 'Blue', 'Red']
#colors = ['grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey']
correlation_handler = Correlation()
mds_handler = MDSHandler()
pca_handler = PCAHandler()
