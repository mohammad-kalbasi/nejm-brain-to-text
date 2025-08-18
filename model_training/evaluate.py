import argparse
import json
import os
import importlib.util

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from rnn_model import GRUDecoder
from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep


def load_lm_modules(lm_path, device, nbest, acoustic_scale):
    """Load the ngram decoder and OPT model from the language_model package."""
    lm_file = os.path.join(os.path.dirname(__file__), "../language_model/language-model-standalone.py")
    spec = importlib.util.spec_from_file_location("lm_standalone", lm_file)
    lm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lm)

    decoder = lm.build_lm_decoder(lm_path, acoustic_scale=acoustic_scale, nbest=nbest)
    opt_model, opt_tokenizer = lm.build_opt(device=device)
    return lm, decoder, opt_model, opt_tokenizer


def main(args):
    device = torch.device(f"cuda:{args.gpu_number}" if torch.cuda.is_available() and args.gpu_number >= 0 else "cpu")

    model_args = OmegaConf.load(os.path.join(args.model_path, "checkpoint/args.yaml"))
    model = GRUDecoder(
        neural_dim=model_args['model']['n_input_features'],
        n_units=model_args['model']['n_units'],
        n_days=len(model_args['dataset']['sessions']),
        n_classes=model_args['dataset']['n_classes'],
        rnn_dropout=model_args['model']['rnn_dropout'],
        input_dropout=model_args['model']['input_network']['input_layer_dropout'],
        n_layers=model_args['model']['n_layers'],
        patch_size=model_args['model']['patch_size'],
        patch_stride=model_args['model']['patch_stride'],
    )
    checkpoint = torch.load(os.path.join(args.model_path, 'checkpoint/best_checkpoint'), weights_only=False)
    for key in list(checkpoint['model_state_dict'].keys()):
        checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)
        checkpoint['model_state_dict'][key.replace('_orig_mod.', '')] = checkpoint['model_state_dict'].pop(key)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # load validation data
    b2txt_csv_df = pd.read_csv(args.csv_path)
    data = {}
    total_trials = 0
    for session in model_args['dataset']['sessions']:
        eval_file = os.path.join(args.data_dir, session, 'data_val.hdf5')
        if os.path.exists(eval_file):
            data[session] = load_h5py_file(eval_file, b2txt_csv_df)
            total_trials += len(data[session]['neural_features'])

    # language model setup
    lm, decoder, opt_model, opt_tokenizer = load_lm_modules(args.lm_path, device, args.nbest, args.acoustic_scale)

    results = []
    for session, sdata in data.items():
        input_layer = model_args['dataset']['sessions'].index(session)
        for i in range(len(sdata['neural_features'])):
            x = sdata['neural_features'][i]
            x = torch.tensor(x, device=device, dtype=torch.bfloat16).unsqueeze(0)
            logits = runSingleDecodingStep(x, input_layer, model, model_args, device)
            log_probs = torch.log_softmax(torch.from_numpy(logits[0]), dim=-1).cpu().numpy()
            lm.lm_decoder.DecodeNumpy(decoder, log_probs, np.zeros_like(log_probs), np.log(args.blank_penalty))
            nbest = decoder.result()
            _, nbest_rescored = lm.gpt2_lm_decode(
                opt_model,
                opt_tokenizer,
                device,
                nbest,
                acoustic_scale=args.acoustic_scale,
                length_penalty=0.0,
                alpha=args.alpha,
            )
            results.append({
                'session': session,
                'block_num': int(sdata['block_num'][i]),
                'trial_num': int(sdata['trial_num'][i]),
                'nbest': nbest_rescored,
            })

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decode validation set with ngram and OPT language models.')
    parser.add_argument('--model_path', type=str, default='../data/t15_pretrained_rnn_baseline')
    parser.add_argument('--data_dir', type=str, default='../data/hdf5_data_final')
    parser.add_argument('--csv_path', type=str, default='../data/t15_copyTaskData_description.csv')
    parser.add_argument('--lm_path', type=str, default='../language_model/pretrained_language_models/openwebtext_1gram_lm_sil')
    parser.add_argument('--gpu_number', type=int, default=0)
    parser.add_argument('--nbest', type=int, default=100)
    parser.add_argument('--acoustic_scale', type=float, default=0.325)
    parser.add_argument('--alpha', type=float, default=0.55)
    parser.add_argument('--blank_penalty', type=float, default=90.0)
    parser.add_argument('--output_json', type=str, default='val_nbest_outputs.json')
    args = parser.parse_args()
    main(args)
