import argparse
from typing import Any
import torchaudio
import nemo.collections.asr as nemo_asr
import torch
import os
from opts import add_decoder_args, add_inference_args
from shutil import rmtree
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel

class ASRTranscription:
    def __init__(self):
        self.audio_file = ''
        self.hook_layers_output_path = os.path.dirname(__file__).replace("\\", "/")
        self.rnn = []
        self.conv = torch.tensor([])
        self.out = torch.tensor([])
        self.enc_output = torch.tensor([])
        self.flag = -1

    def store_output(self, tensor, dump_folder):
        self.flag += 1
        os.makedirs(dump_folder, exist_ok=True)
        fname = len(os.listdir(dump_folder))
        torch.save(tensor, os.path.join(dump_folder, f"{fname}.pt"))

        if self.flag == 0:
            self.enc_output = tensor
        elif self.flag in [1, 2, 3, 4, 5]:
            self.rnn.append(tensor)
        elif self.flag == 6:
            self.conv = tensor
        elif self.flag == 7:
            self.out = tensor

    def intermediate_hook(self, dump_folder):
        def hook(module, input, output):
            self.store_output(output.detach().cpu().transpose(1, 2), dump_folder)
        return hook

    def process_audio_file(self):
        waveform, sample_rate = torchaudio.load(self.audio_file)
#####################################################################
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_de_conformer_ctc_large")
        # using the asr nemo lang model 
        model = EncDecRNNTBPEModel.from_pretrained(model_name="stt_de_conformer_transducer_large")
        model.decoder
#####################################################################
        wav = waveform.mean(dim=0)
        wav = wav.unsqueeze(0)
        wav_len = torch.tensor(len(wav[0])).unsqueeze(0)

        empty_tensor = torch.tensor([])

        
        for f in range(len(asr_model.encoder.layers) - 1):
            if f == 5:
                break
            asr_model.encoder.layers[f].register_forward_hook(
                self.intermediate_hook(os.path.join(self.hook_layers_output_path, f"b{f}"))
            )

        asr_model.encoder.pre_encode.conv.register_forward_hook(
            self.intermediate_hook(os.path.join(self.hook_layers_output_path, "conv"))
        )

        asr_model.encoder.pre_encode.out.register_forward_hook(
            self.intermediate_hook(os.path.join(self.hook_layers_output_path, "out"))
        )

        last_dumpdir = os.path.join(self.hook_layers_output_path, "enco")
        
        ################################################################################### gathering the decoder out, output_sizes
        # model.decoder.pre_encode.out.register_forward_hook(
        # self.intermediate_hook(os.path.join(self.hook_layers_output_path, "decoder_out"))
        # )

        # last_dumpdir = os.path.join(self.hook_layers_output_path, "decoder_enco")
        ###################################################################################
 

        with torch.no_grad():
            processed, processed_len = asr_model.preprocessor(input_signal=wav, length=wav_len)
            enc_out = asr_model.encoder(audio_signal=processed, length=processed_len)[0].detach().cpu()
            #####
 
            # dec_out = asr_model.decoder( encoder_output=enc_out) 
        self.store_output(enc_out, last_dumpdir)
 
         
        print("below are the encoder output")
        print(self.enc_output)

        subsampling_factor = getattr(asr_model.encoder.pre_encode, 'subsample_factor', 4)
        encoder_output_lengths = torch.ceil(wav_len.float() / subsampling_factor).int()
        output_sizes = encoder_output_lengths // subsampling_factor

        global output_sizes_forward
        output_sizes_forward = output_sizes.detach()
        output_sizes_forward = output_sizes.float().requires_grad_()
        # print("Output Sizes:", output_sizes)

        print("below are the rnn")
        print(len(self.rnn))
        global rnn_0_forward, rnn_1_forward, rnn_2_forward, rnn_3_forward, rnn_4_forward
        rnn_0_forward = self.rnn[0].float().requires_grad_()
        rnn_1_forward = self.rnn[1].float().requires_grad_()
        rnn_2_forward = self.rnn[2].float().requires_grad_()
        rnn_3_forward = self.rnn[3].float().requires_grad_()
        rnn_4_forward = self.rnn[4].float().requires_grad_()

        print("below are the conv")
        global con_forward
        con_forward = self.conv.float().requires_grad_()
        print(con_forward)

        print("below are the out")
        global out_forward
        out_forward = self.out.float().requires_grad_()
        print(out_forward)

 



        '''
        # This line is to use the stt_de_conformer_transducer_large Language Model from Nvidia:
        
        from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel

        model = EncDecRNNTBPEModel.from_pretrained(model_name="stt_de_conformer_transducer_large")
        model.decoder


        # ####################################################

        # Acoustic model
        import nemo.collections.asr as nemo_asr

        asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_de_conformer_transducer_large")

        '''

        print('********************** Done! ***********************')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSpeech transcription')
    parser = add_inference_args(parser)
    parser = add_decoder_args(parser)
    args = parser.parse_args()

    asr_transcription = ASRTranscription()
    # Enter the path to the audio file (WAV format)
    audio_file_path='/home/mmm2050/QU_DFKI_Thesis/Experimentation/ASR_Accent_Analysis_De/audio/wav/audio_De_08042023/common_voice_de_30676740.wav' 

    asr_transcription.audio_file = audio_file_path

    asr_transcription.process_audio_file()

