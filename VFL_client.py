import time
import grpc
import argparse
from core.ActiveParty import ActiveParty


def create_client_credentials(server_cert_path, server_key_path, root_ca_cert_path):

    client_credentials = grpc.ssl_channel_credentials(
        root_certificates=open(root_ca_cert_path, 'rb').read(),
        private_key=open(server_key_path, 'rb').read(),
        certificate_chain=open(server_cert_path, 'rb').read()
    )
    return client_credentials


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--congig', dest='config_filepath', help='config filepath', default='config/config.conf', type=str)
    parser.add_argument('-m', '--mod', dest='mod', help='One from list \'linear\', \'logistic\', \'softmax\'', type=str,
                        default='softmax')
    parser.add_argument('-t', '--trainPath', dest='trainPath', help='Path to .csv with train dataset', type=str,
                        default=r"data/active_party_train.csv")
    parser.add_argument('-v', '--validPath', dest='validPath', help='Path to .csv with valid dataset', type=str,
                        default=r"data/active_party_test.csv")
    parser.add_argument('-d', '--id-col', dest='id_col', help='Name of ID column in the datasets', type=str, default='ID')
    parser.add_argument('-y', '--target', dest='target', help='Name of Target column in the datasets', type=str, default='y')
    parser.add_argument('-f', '--frac', dest='frac', help='Proportion of data randomly selected from the datasets',
                        type=float, default=0.5)
    parser.add_argument('-ns', '--nosecure', dest='secure_channel', help='Communication channel encryption flag',
                        action='store_false', default=True)
    parser.add_argument('-cp', '--cert_path', dest='cert_path', help='Path to .crt with server certificate',
                        type=str, default='cert/127.0.0.1.crt')
    parser.add_argument('-kp', '--key_path', dest='key_path', help='Path to .key with server key',
                        type=str, default='cert/127.0.0.1.key')
    parser.add_argument('-rp', '--root_path', dest='root_path', help='Path to .crt with server root certificate',
                        type=str, default='cert/rootCA.crt')

    args = parser.parse_args()

    mod = args.mod

    creds = None
    if args.secure_channel:
        creds = create_client_credentials(server_cert_path=args.cert_path, server_key_path=args.key_path,
                                          root_ca_cert_path=args.root_path)

    ap = ActiveParty(mod=mod, config_path=args.config_filepath, credentials=creds)
    ap.load_dataset(args.trainPath, args.validPath, args.id_col, args.target, args.frac)

    t = time.time()
    ap.train()
    print('Training time = ', time.time() - t)

    file_name = ap.dump_model(r'res_models/party0')

    ap.load_model(file_name)
    ap.predict()
