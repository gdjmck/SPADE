import pyproj

def longtitude_to_coord(lon, lat):
    """

    :param lon: 经度
    :param lat: 纬度
    :return: x, y坐标
    """
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    return transformer.transform(lon, lat)

def coord_to_longtitude(x, y):
    """

    :param x: 坐标x值
    :param y: 坐标y值
    :return: 经度, 纬度
    """
    transformer = pyproj.Transformer.from_crs('EPSG:3857', 'EPSG:4326')
    return transformer.transform(x, y)