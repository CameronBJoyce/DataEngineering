"""
Export the results of an inference process to the Elastic Stack
"""

__author__ = """
Cameron Joyce
#### Charleston, SC
"""

"""
Steps To Run:
1. Configure elasticsearch index (name, port, etc.) in config.yaml and config.py
2. Start Docker
3. Launch docker-compose file (Run: docker-compose up)
4. Run program (i.e. minimaltest.ipynb)
5. Data will be indexed at elasticsearch input
"""
__all__ = []

# **************************************************

import sys
import LogUtilities as lu
from datetime import datetime
from elasticsearch import Elasticsearch, helpers

# **************************************************

class ExportToElasticsearch(object):
    """
    Export Inference Results to Elastic Stack
    """

    # **********************************************

    def __init__(self):
        """
        Initialize the output processor
        """

    # **************************************************

    def init(self, config):
        """
        Object initialization independent of construction
        """
        self._netaddr = config.elasticsearch1.netaddr #net address format = [localhost:443]
        self._es = Elasticsearch(self._netaddr)
        self._class_totals = [0] * 2
        self._index_name = config.elasticsearch1.index_name
        self._timeout = config.elasticsearch1.timeout
        
    # **************************************************

    def __call__(self, out_vec):
        """
        Process a batch
        """
        d = {}
        dictList = []
    
        for result in out_vec:
            time1 = result[1]['startTime']
            time2 = result[1]['endTime']
            d['Cls0'] = float(result[0][0])
            d['Cls1'] = float(result[0][1])
            d['sourceIP'] = result[1]['sourceIP']
            d['destinationIP'] = result[1]['destinationIP']
            d['startTime'] = time1[:25]
            d['endTime'] = time2[:25]
            dictList.append(dict(d.copy()))
            
            self._class_totals[0 if result[0][0] > result[0][1] else 1] += 1
       
        helpers.bulk(self._es, dictList, index=self._index_name, request_timeout=self._timeout)

    # **************************************************
    def term(self):
        """
        Object destruction indepentent of desrruction

        Print and return the summary
        """

        lu.info(f"{self.__class__.__name__} Summary:")

        [lu.log_value(f"Class {i} total", "{}", total) for i, total in enumerate(self._class_totals)]

        return dict(by_class={cls: total for cls, total in enumerate(self._class_totals)})

# **************************************************

if __name__ == '__main__':
     sys.exit("Not a main program")
       
