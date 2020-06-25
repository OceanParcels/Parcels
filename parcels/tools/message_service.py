from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection, wait
from os import getpid
from time import sleep
import sys
from parcels.tools import logger


try:
    from mpi4py import MPI
except:
    MPI = None


def execute_requested_messages(mpi_proc_connections, exec_class):
    """

    """
    assert (isinstance(mpi_proc_connections, list))
    print(mpi_proc_connections)
    assert (len(mpi_proc_connections) > 0)
    assert (isinstance(mpi_proc_connections[0], Connection))
    requester_obj = exec_class()
    while True:
        # for _request_proc_ in mpi_proc_connections:
        #     _request_proc_ = Connection()
        ready_procs = wait(mpi_proc_connections, -1)
        if len(ready_procs) > 0:
            for _request_proc_ in ready_procs:
                func_name = _request_proc_.recv()
                args = int(_request_proc_.recv())
                argv = []
                if args > 0:
                    argv = _request_proc_.recv()
                while _request_proc_.recv() != '#e#':
                    pass

                call_func = getattr(requester_obj, func_name)
                res = None
                if call_func is not None:
                    if args > 0:
                        res = call_func(*argv)
                    else:
                        res = call_func()
                if res is not None:
                    if not _request_proc_.closed:
                        _request_proc_.send(res)
                        _request_proc_.send('#e#')
        else:
            sleep(0.2)


def mpi_pingpong(req_tag = 5, resp_tag = 6):
    """

    """
    while True:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        msg = mpi_comm.irecv(source=MPI.ANY_SOURCE, tag=req_tag)
        data = msg.wait()
        assert isinstance(data, dict)
        dst = int(data["src_rank"])
        # print(data)
        sys.stdout.write("{}\n".format(data))
        data_package = {"result": 10, "src_rank": mpi_rank}
        mpi_comm.send(data_package, dest=dst, tag=resp_tag)
        sleep(0.1)


def mpi_execute_requested_messages(exec_class, request_tag = 0, response_tag = 1):
    """

    """
    requester_obj = exec_class()
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    logger.info("service - MPI rank: {} pid: {}".format(mpi_rank, getpid()))
    _subscribed = {}
    _running = True
    while _running:
        msg_status = MPI.Status()
        msg = mpi_comm.irecv(source=MPI.ANY_SOURCE, tag=request_tag)
        test_result = msg.test(status=msg_status)
        #while not (isinstance(test_result, tuple) or isinstance(test_result, list)) or ((test_result[0] == False) or (test_result[0] == True and not isinstance(test_result[1], dict))):
        while (test_result[0] == False) or (test_result[0] == True and not isinstance(test_result[1], dict)):
            test_result = msg.test(status=msg_status)

        request_package = test_result[1]
        # logger.info("ID serv. - recv.: {} - (srv. rank: {}; snd. rank: {}; pkg. rank: {}".format(request_package["func_name"], mpi_rank, msg_status.Get_source(), request_package["src_rank"]))
        assert isinstance(request_package, dict)

        func_name = request_package["func_name"]
        args = int(request_package["args"])
        argv = []
        if args > 0:
            argv = request_package["argv"]

        if func_name == "subscribe":
            logger.info("'subscribe' message received.")
            _subscribed[msg_status.Get_source()] = True
        elif func_name == "abort":
            # logger.info("'abort' message received (src: {}).".format(msg_status.Get_source()))
            _subscribed[msg_status.Get_source()] = False
            logger.info("Subscribers: {}".format( _subscribed ))
            _running = False
            for flag in _subscribed:
                _running |= flag
            if not _running:
                break
        else:
            call_func = getattr(requester_obj, func_name)
            res = None
            if call_func is not None:
                if args > 0:
                    res = call_func(*argv)
                else:
                    res = call_func()
            if res is not None:
                response_package = {"result": res, "src_rank": mpi_rank}
                # logger.info("sending message: {}".format(response_package))
                mpi_comm.send(response_package, dest=msg_status.Get_source(), tag=response_tag)

    logger.info("ABORTED ID Service")