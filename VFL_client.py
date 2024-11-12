import argparse
from core.ActiveParty import ActiveParty


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--congig', dest='config_filepath', help='config filepath', default='config/config.conf', type=str)
    parser.add_argument('-m', '--mod', dest='mod', help='One from list \'linear\', \'logistic\', \'softmax\'', type=str, default='softmax')
    parser.add_argument('-t', '--trainPath', dest='trainPath', help='Path to .csv with train dataset', type=str,  default=r"data/bank_train.csv")
    parser.add_argument('-v', '--validPath', dest='validPath', help='Path to .csv with valid dataset', type=str,  default=r"data/bank_test.csv")
    parser.add_argument('-d', '--id-col', dest='id_col', help='Name of ID column in the datasets', type=str, default='ID')
    parser.add_argument('-y', '--target', dest='target', help='Name of Target column in the datasets', type=str, default='y')
    parser.add_argument('-f', '--frac', dest='frac', help='Proportion of data randomly selected from the datasets', type=float, default=0.5)
    args = parser.parse_args()

    ap = ActiveParty(mod=args.mod, config_path=args.config_filepath)
    ap.load_dataset(args.trainPath, args.validPath, args.id_col, args.target, args.frac)

    ap.train()
    file_name = ap.dump_model(r'res_models/party0')

    ap.load_model(file_name)
    ap.predict()
