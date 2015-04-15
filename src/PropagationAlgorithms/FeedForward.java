package PropagationAlgorithms;

import java.util.ArrayList;
import java.util.List;

import ActivationFunctions.ActivationFunction;
import NeuralNet.NeuralNet;

public class FeedForward {
	private NeuralNet neuralNetwork;
	private ActivationFunction activationFunction;
	private List<Double> inputSet;
	private List<Double> netHidden;

	public FeedForward(NeuralNet n,ActivationFunction a){
		this.neuralNetwork = n;
		this.activationFunction = a;
		netHidden = new ArrayList<Double>();
	}
	public void setInputSet(List<Double> input){
		this.inputSet = input;
	}
	public Double getNetHidden(Integer i){
		return this.netHidden.get(i);
	}
	public void setNeuralNetwork(NeuralNet n){
		neuralNetwork = n;
	}
	public ArrayList<Double> executeAndReturnResult(List<Double> input){
		setInputSet(input);
		
		ArrayList<Double> netOutput = new ArrayList<Double>();
		
		//computing net hidden
		for(int j = 0 ; j < neuralNetwork.getNumberOfHiddenNodes();++j){
			Double netHiddenJ = 0.0;
			for(int i=0;i<neuralNetwork.getNumberOfInputNodes();++i){
				netHiddenJ +=
					neuralNetwork.getInputHiddenWeight(i, j)
					*
					inputSet.get(i);
			}
			this.netHidden.add(activationFunction.activate(netHiddenJ + neuralNetwork.getHiddenBias(j)));
		}

		//computing net output
		for(int k=0;k<neuralNetwork.getNumberOfOutputNodes();++k){
			Double netOutputK = 0.0;
			for(int j=0;j<neuralNetwork.getNumberOfHiddenNodes();j++){
				netOutputK +=
						neuralNetwork.getHiddenOutputWeight(j, k)
						*
						netHidden.get(j);
			}
			netOutput.add(activationFunction.activate(netOutputK + neuralNetwork.getOutputBias(k)));
		}
		
		return netOutput;
	}

}
