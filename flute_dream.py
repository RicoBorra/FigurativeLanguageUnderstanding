from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

def add_combined_cols(entry):
    
    premise = entry["premise"].strip()
    hypothesis = entry["hypothesis"].strip()
    
    if not premise.endswith("."):
        premise += "."
    assert(premise.endswith("."))
    if not hypothesis.endswith("."):
        hypothesis += "."
    assert(hypothesis.endswith("."))
    
    # Columns for System 1
    entry["premise_hypothesis"] = 'Premise: ' + premise + ' Hypothesis: ' + hypothesis + ' Is there a contradiction or entailment between the premise and hypothesis ?'
    entry["label_explanation"] = 'Label: ' + entry["label"] + '. Explanation: ' + entry["explanation"]

    # Columns for System 2
    entry["premise_hypothesis_system_2"] = 'Premise: ' + premise + ' Hypothesis: ' + hypothesis + ' What is the type of figurative language involved? Is there a contradiction or entailment between the premise and hypothesis ?'
    entry["type_label_explanation"] = 'Type: ' + entry["type"] + '. Label: ' + entry["label"] + '. Explanation: ' + entry["explanation"]
    
    # Columns for Systems 3
    for dream_dimension in ['emotion', 'motivation', 'consequence', 'rot'] :
        entry["premise_hypothesis_" + dream_dimension] = 'Premise: ' + premise + ' [' + dream_dimension.capitalize() + '] ' + entry['premise_' + dream_dimension].strip() + \
                    ' Hypothesis: ' + hypothesis + ' [' + dream_dimension.capitalize() + '] ' + entry['hypothesis_' + dream_dimension] + ' Is there a contradiction or entailment between the premise and hypothesis ?'
    entry["premise_hypothesis_all_dims"] = 'Premise: ' + premise + \
                ' [Emotion] ' + entry['premise_emotion'].strip() + \
                ' [Motivation] ' + entry['premise_motivation'].strip() + \
                ' [Consequence] ' + entry['premise_consequence'].strip() + \
                ' [Rot] ' + entry['premise_rot'].strip() + \
                ' Hypothesis: ' + hypothesis + \
                ' [Emotion] ' + entry['hypothesis_emotion'].strip() + \
                ' [Motivation] ' + entry['hypothesis_motivation'].strip() + \
                ' [Consequence] ' + entry['hypothesis_consequence'].strip() + \
                ' [Rot] ' + entry['hypothesis_rot'].strip()
    
    # Columns for System 4 (For the explanation part)
    '''As the specific input isn't indicated in the paper, the question tries to formalize at best what is expected'''
    entry["premise_hypothesis_label"] = 'Premise: ' + premise + ' Hypothesis: ' + hypothesis + ' Label : ' + entry['label'] + '. What is the explanation of the label associated to the premise and the hypothesis ?'
    
    return entry

'''Class encapsulating the two steps of System 4 (Classify, then Explain)'''
class DREAM_FLUTE_System4 :
    def __init__(self, tokenizer = None, model_s41_path = None, model_s42_path = None) -> None:
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained("t5-small")
        self.model_s41 = AutoModelForSeq2SeqLM.from_pretrained(model_s41_path) if model_s41_path is not None else AutoModelForSeq2SeqLM.from_pretrained("YoanBOUTE/DREAM-FLUTE-S4-Classify")
        self.model_s42 = AutoModelForSeq2SeqLM.from_pretrained(model_s42_path) if model_s42_path is not None else AutoModelForSeq2SeqLM.from_pretrained("YoanBOUTE/DREAM-FLUTE-S4-Explain")

    '''Expected input for function : "Premise : ... . Hypothesis : ... . Is there a contradiction or entailment between the premise and hypothesis ?" 
    Or list of strings in this format'''
    def prediction_pipeline(self, inputs) :
        if isinstance(inputs, str) :
            tok_input = self.tokenizer(inputs, return_tensors='pt').input_ids
            output_model_1 = self.model_s41.generate(tok_input, max_new_tokens=100)
            decoded_output_model_1 = "Label : " + self.tokenizer.decode(output_model_1[0], skip_special_tokens=True)
            intermediate_input = inputs[:inputs.find("Is there a contradiction or entailment between the premise and hypothesis ?")] + decoded_output_model_1 + ". What is the explanation of the label associated to the premise and the hypothesis ?"
            tok_intermediate_input = self.tokenizer(intermediate_input, return_tensors='pt').input_ids
            output_model_2 = self.model_s42.generate(tok_intermediate_input, max_new_tokens=100)

            return decoded_output_model_1 + ". Explanation : " + self.tokenizer.decode(output_model_2[0], skip_special_tokens=True)
        
        elif isinstance(inputs, list) and all(isinstance(input, str) for input in inputs) :
            predictions = []
            for input in inputs :
                predictions.append(self.prediction_pipeline(input))
            
            return predictions
        
        else :
            raise TypeError('Inputs should be either a list of two strings or a list of lists of two strings')
        

'''Ensemble class that loads all models from HuggingFace (or from the device if a path to the model is indicated) 
and implements the ensembling algorithm given in the DREAM-FLUTE paper'''
class DREAM_FLUTE_Ensemble :
    def __init__(self, tokenizer_path = None, s1_path = None, s2_path = None,
                 s3_emo_path = None, s3_mot_path = None, s3_cons_path = None,
                 s3_rot_path = None, s3_alldims_path = None, s4_clas_path = None, 
                 s4_exp_path = None, dream_path = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) if tokenizer_path is not None else AutoTokenizer.from_pretrained("t5-small")
        self.model_s1 = AutoModelForSeq2SeqLM.from_pretrained(s1_path) if s1_path is not None else AutoModelForSeq2SeqLM.from_pretrained("YoanBOUTE/DREAM-FLUTE-S1")
        self.model_s2 = AutoModelForSeq2SeqLM.from_pretrained(s2_path) if s2_path is not None else AutoModelForSeq2SeqLM.from_pretrained("YoanBOUTE/DREAM-FLUTE-S2")
        self.model_s3_emo = AutoModelForSeq2SeqLM.from_pretrained(s3_emo_path) if s3_emo_path is not None else AutoModelForSeq2SeqLM.from_pretrained("YoanBOUTE/DREAM-FLUTE-S3-Emotion")
        self.model_s3_mot = AutoModelForSeq2SeqLM.from_pretrained(s3_mot_path) if s3_mot_path is not None else AutoModelForSeq2SeqLM.from_pretrained("YoanBOUTE/DREAM-FLUTE-S3-Motivation")
        self.model_s3_cons = AutoModelForSeq2SeqLM.from_pretrained(s3_cons_path) if s3_cons_path is not None else AutoModelForSeq2SeqLM.from_pretrained("YoanBOUTE/DREAM-FLUTE-S3-Consequence")
        self.model_s3_rot = AutoModelForSeq2SeqLM.from_pretrained(s3_rot_path) if s3_rot_path is not None else AutoModelForSeq2SeqLM.from_pretrained("YoanBOUTE/DREAM-FLUTE-S3-ROT")
        self.model_s3_alldims = AutoModelForSeq2SeqLM.from_pretrained(s3_alldims_path) if s3_alldims_path is not None else AutoModelForSeq2SeqLM.from_pretrained("YoanBOUTE/DREAM-FLUTE-S3-AllDims")
        self.model_s4 = DREAM_FLUTE_System4(self.tokenizer, s4_clas_path, s4_exp_path)
        self.model_dream = AutoModelForSeq2SeqLM.from_pretrained(dream_path) if dream_path is not None else AutoModelForSeq2SeqLM.from_pretrained("RicoBorra/DREAM-t5-small")
    
    '''Tokenizes the input, then feeds it to the given model, and decodes the output to have a string as result.
    This method is callable for all models except System 4 (Use the method defined in the class of System 4)'''
    def _prediction_pipeline(self, input : str, model) -> str :
        tokenized_input = self.tokenizer(input, return_tensors='pt').input_ids
        model_output = model.generate(tokenized_input, max_new_tokens=100)
        decoded_output = self.tokenizer.decode(model_output[0], skip_special_tokens=True)
        return decoded_output
    
    '''Preprocesses the input for each model, then feeds it to the pipeline.
    Returns a dictionary of all models' predictions.'''
    def _get_all_predictions(self, input : list) :
        prem, hyp = input 
        prem = prem.strip()
        hyp = hyp.strip()
        if not prem.endswith('.') :
            prem += '.'
        if not hyp.endswith('.') :
            hyp += '.' 

        predictions = dict()

        input_1 = f"Premise : {prem} Hypothesis : {hyp} Is there a contradiction or entailment between the premise and hypothesis ?"
        predictions['S1'] = self._prediction_pipeline(input_1, self.model_s1)

        input_2 = f"Premise : {prem} Hypothesis : {hyp} What is the type of figurative language involved? Is there a contradiction or entailment between the premise and hypothesis ?"
        predictions['S2'] = self._prediction_pipeline(input_2, self.model_s2)

        # DREAM elaborations for system 3
        input_dream_prem = f"[SITUATION] {prem} [QUERY] "
        input_dream_hyp = f"[SITUATION] {hyp} [QUERY] "
        prem_elaborations = {key : self._prediction_pipeline(input_dream_prem + key, self.model_dream) for key in ['emotion', 'motivation', 'consequence', 'rot']}
        for key, elab in prem_elaborations.items() :
            elab = elab.strip()
            if not elab.endswith('.') :
                prem_elaborations[key] += '.' 
        hyp_elaborations = {key : self._prediction_pipeline(input_dream_hyp + key, self.model_dream) for key in ['emotion', 'motivation', 'consequence', 'rot']}
        for key, elab in hyp_elaborations.items() :
            elab = elab.strip()
            if not elab.endswith('.') :
                hyp_elaborations[key] += '.' 

        input_3_emo = f"Premise : {prem} [Emotion] {prem_elaborations['emotion']} Hypothesis : {hyp} [Emotion] {hyp_elaborations['emotion']} Is there a contradiction or entailment between the premise and hypothesis ?"
        predictions['S3-emo'] = self._prediction_pipeline(input_3_emo, self.model_s3_emo)

        input_3_mot = f"Premise : {prem} [Motivation] {prem_elaborations['motivation']} Hypothesis : {hyp} [Motivation] {hyp_elaborations['motivation']} Is there a contradiction or entailment between the premise and hypothesis ?"
        predictions['S3-mot'] = self._prediction_pipeline(input_3_mot, self.model_s3_mot)

        input_3_cons = f"Premise : {prem} [Consequence] {prem_elaborations['consequence']} Hypothesis : {hyp} [Consequence] {hyp_elaborations['consequence']} Is there a contradiction or entailment between the premise and hypothesis ?"
        predictions['S3-cons'] = self._prediction_pipeline(input_3_cons, self.model_s3_cons)

        input_3_rot = f"Premise : {prem} [Rot] {prem_elaborations['rot']} Hypothesis : {hyp} [Rot] {hyp_elaborations['rot']} Is there a contradiction or entailment between the premise and hypothesis ?"
        predictions['S3-rot'] = self._prediction_pipeline(input_3_rot, self.model_s3_rot)

        input_3_all = f"Premise : {prem} "
        for key, elab in prem_elaborations.items() :
            input_3_all += f"[{key.capitalize()}] {elab} "
        input_3_all += f"Hypothesis : {hyp} "
        for key, elab in hyp_elaborations.items() :
            input_3_all += f"[{key.capitalize()}] {elab} "
        input_3_all += "Is there a contradiction or entailment between the premise and hypothesis ?"
        predictions['S3-all'] = self._prediction_pipeline(input_3_all, self.model_s3_alldims)

        # The input for system 4 is in the same format as for system 1
        predictions['S4'] = self.model_s4.prediction_pipeline(input_1)

        return predictions
    
    '''Uses the predictions from each model to compute the final prediction of the ensemble'''
    def _ensemble_algorithm(self, model_outputs) :
        # Firstly, the label is selected based on the majority between the 5 best models (according to the paper : systems 1, 2, 3-motivation, 3-alldims, 4)
        labels = [model_outputs[key].split('.')[0] for key in ['S1', 'S2', 'S3-mot', 'S3-all', 'S4']]
        # Sometimes, it might happen with the small models that the generated label is a mix of words, like 'Contratailment' or 'Endiction'
        for label in labels :
            if label not in ['Label : Contradiction', 'Label : Entailment'] :
                labels.remove(label)
        unique, counts = np.unique(labels, return_counts=True)
        ix = np.argmax(counts)
        major_label = unique[ix]

        # Then, pick the explanation of the first system agreeing with the majority label, following an order indicated in the paper
        for key in ['S3-cons', 'S3-emo', 'S2', 'S3-all', 'S3-mot', 'S4', 'S1'] :
            substrings = model_outputs[key].split('.')
            label = substrings[0]
            explanation = substrings[1]

            if label == major_label :
                break

        return major_label + '.' + explanation + '.'

    '''Expected input : [Premise_sentence, hypothesis_sentence] or list of inputs'''
    def predict(self, inputs) :
        if isinstance(inputs, list) and all(isinstance(input, str) for input in inputs) and len(inputs) == 2 :
            preds = self._get_all_predictions(inputs)
            final_pred = self._ensemble_algorithm(preds)
        
            return final_pred 
        
        elif isinstance(inputs, list) and all(isinstance(input, list) for input in inputs) :
            predictions = []
            for input in inputs :
                predictions.append(self.predict(input))
            
            return predictions
        else :
            raise TypeError('Inputs should be either a list of two strings or a list of lists of two strings')