# -*- coding:utf-8 -*-
# @Time: 2022/8/25 15:34
# @File: config.py
# @Software: PyCharm

import yaml, os

# from icecream import ic

# cfg.yaml 配置文件内容
'''
database:
  ip: 192.168.2.1
  port: 8080
redis:
  ip: 192.168.2.2
  port: 6379
host:
  group:
    name: qqq
    passwd: 333
'''


class DictToClass(object):
    '''
    ;;将字典准换为 class 类型
    '''

    @classmethod
    def _to_class(cls, _obj):
        _obj_ = type('new', (object,), _obj)
        [setattr(_obj_, key, cls._to_class(value)) if isinstance(value, dict) else setattr(_obj_, key, value) for
         key, value in _obj.items()]
        return _obj_


class ReadConfigFiles(object):
    def __init__(self):
        '''
        ;;获取当前工作路径
        '''
        self.works_path = os.path.dirname(os.path.realpath(__file__))

    @classmethod
    def open_file(cls, path):
        '''
        ;;读取当前工作目录下cfg.yaml文件内容，并返回字典类型
        :return:
        '''
        return yaml.load(
            open(path, 'r', encoding='utf-8').read(), Loader=yaml.FullLoader
        )

    @classmethod
    def cfg(cls, path, item=None):
        '''
        ;;调用该方法获取需要的配置，item如果为None，返回则是全部配置
        :param item:
        :return:

        Args:
            path:
        '''
        return DictToClass._to_class(cls.open_file(path).get(item) if item else cls.open_file(path))


if __name__ == '__main__':
    '''
    ;;调用ReadConfigFiles.cfg('传递参数 or null')
    '''
    cfg = ReadConfigFiles.cfg()
    # 测试输出
    # ic(cfg.host.group)
