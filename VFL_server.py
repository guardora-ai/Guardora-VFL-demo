from concurrent import futures

import grpc
import proto_compiled.service_pb2_grpc as pb2_grpc

import os
import shutil
import argparse
from core.PassiveParty import PassiveParty


class MVP_VFL(pb2_grpc.MVP_VFLServicer):
    def __init__(self, pp) -> None:
        self.passive_party = pp

    def request_analysis(self, request, context):
        if request.processor == 'recvActivePubKey':
            return self.passive_party.recv_active_pub_key(request.pub_key)
        elif request.processor == 'recvPSIActivePubKey':
            return self.passive_party.recv_active_psi_pub_key(request.pub_key)
        elif request.processor == 'recvPSISaltedSamples':
            return self.passive_party.recv_psi_salted_samples()
        elif request.processor == 'encPSISaltSamplesSEND':
            return self.passive_party.psi_intersect_samples(request)
        elif request.processor == 'recvPSIfindCOMMONsamples':
            return self.passive_party.psi_recv_common_samples()
        elif request.processor == 'sendPSIFINALsamples':
            return self.passive_party.psi_instanse_sample(request)
        elif request.processor == 'initSampleAlign':
            return self.passive_party.sample_align()
        elif request.processor == 'recvSampleList':
            return self.passive_party.recv_sample_list(request)
        elif request.processor == 'initWeightedData':
            return self.passive_party.calc_weighted_data(request)
        elif request.processor == 'recvGradientsReg':
            return self.passive_party.calc_salt_grad(request)
        elif request.processor == 'recvDecryptedGradients':
            return self.passive_party.calc_updates_for_weight(request)
        elif request.processor == 'recvDumpModel':
            return self.passive_party.dump_weight(request.model_name)
        elif request.processor == 'recvGradients':
            return self.passive_party.splits_sum(request)
        elif request.processor == 'confirmSplit':
            return self.passive_party.confirm_split(request)
        elif request.processor == 'recvLoadModel':
            return self.passive_party.load_weight(request.model_name)
        elif request.processor == 'recvPredict':
            return self.passive_party.predict(request)
        else:
            return self.passive_party.empty_res()


def serve(party, port, secure_channel, serv_cert_path, serv_key_path):

    size_params = [('grpc.max_send_message_length', 512 * 1024 * 1024),
                   ('grpc.max_receive_message_length', 512 * 1024 * 1024)]

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=size_params)
    pb2_grpc.add_MVP_VFLServicer_to_server(MVP_VFL(party), server)

    if secure_channel:
        server_credentials = grpc.ssl_server_credentials([(open(serv_key_path, 'rb').read(),
                                                           open(serv_cert_path, 'rb').read())])
        server.add_secure_port("[::]:" + str(port), server_credentials=server_credentials)
        secure = 'secure'
    else:
        server.add_insecure_port("[::]:" + str(port))
        secure = 'insecure'

    server.start()
    print(f"Server started, listening on {secure} port " + str(port))
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', '--id', dest='id', help='id for this party', type=int, default=1)
    parser.add_argument('-p', '--port', dest='port', help='port for this party', type=int, default=50055)
    parser.add_argument('-t', '--trainPath', dest='trainPath', help='Path to .csv with train dataset', type=str,
                        default=r"data/passive_party_train.csv")
    parser.add_argument('-v', '--validPath', dest='validPath', help='Path to .csv with valid dataset', type=str,
                        default=r"data/passive_party_test.csv")
    parser.add_argument('-d', '--id-col', dest='id_col', help='Name of ID column in the datasets', type=str,
                        default='ID')
    parser.add_argument('-f', '--frac', dest='frac', help='Proportion of data randomly selected from the datasets',
                        type=float, default=1.0)
    parser.add_argument('-ns', '--nosecure', dest='secure_channel', help='Communication channel encryption flag',
                        action='store_false', default=True)
    parser.add_argument('-cp', '--cert_path', dest='cert_path', help='Path to .crt with server certificate',
                        type=str, default='cert/127.0.0.1.crt')
    parser.add_argument('-kp', '--key_path', dest='key_path', help='Path to .key with server key',
                        type=str, default='cert/127.0.0.1.key')

    args = parser.parse_args()

    id = args.id
    port = args.port

    path_list = [
        r'temp/file/party-' + str(id),
        r'temp/model/party-' + str(id)
    ]

    for path in path_list:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    pp = PassiveParty(id)
    pp.path_list = path_list
    pp.load_dataset(args.trainPath, args.validPath, args.id_col, args.frac)
    pp.set_dump_path(r'res_models/party' + str(id))

    serve(pp, port, secure_channel=args.secure_channel, serv_cert_path=args.cert_path, serv_key_path=args.key_path)
