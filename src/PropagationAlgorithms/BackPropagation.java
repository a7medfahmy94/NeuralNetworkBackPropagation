package PropagationAlgorithms;

import java.util.ArrayList;

import ActivationFunctions.Sigmoid;
import NeuralNet.NeuralNet;

public class BackPropagation {
	private NeuralNet neuralNetwork;
	private FeedForward feedForward;
	private ArrayList<ArrayList<Double> > trainingInputs;
	private ArrayList<ArrayList<Double> > trainingOutputs;
	private Double totalMSE;
	private Integer numberOfIterationsPerTrainingSet;
	private Double learningRate;
	
	
	public BackPropagation(NeuralNet nn,Integer numOfIterations,double gamma){
		this.neuralNetwork = nn;
		this.feedForward = new FeedForward(nn, new Sigmoid(gamma));
		this.totalMSE = 0.0;
		this.learningRate = 1.0;
		this.numberOfIterationsPerTrainingSet = numOfIterations;
	}
	
	public void setTrainingExamples(ArrayList<ArrayList<Double> > ins,
			ArrayList<ArrayList<Double> > out){
		this.trainingInputs = ins;
		this.trainingOutputs = out;		
	}
	public NeuralNet getNeuralNet(){
		return this.neuralNetwork;
	}
	public void setLearningRate(Double d){
		this.learningRate = d;
	}
	public Double getTotalMSE(){
		return this.totalMSE;
	}
	
	public void executeBackPropagation(ArrayList<ArrayList<Double> > ins,
			ArrayList<ArrayList<Double> > out){
		
		
		
		setTrainingExamples(ins, out);
		double mnError = 1000000.0;
		NeuralNet bestYet = new NeuralNet(neuralNetwork.getNumberOfInputNodes(), neuralNetwork.getNumberOfHiddenNodes(), neuralNetwork.getNumberOfOutputNodes());

		//iterate over the examples many times
		for(int globalIterationCount = 0;globalIterationCount < this.numberOfIterationsPerTrainingSet;++globalIterationCount){
			double errorPerIteration = 0;
			//for each example
			for(int trainingExampleI = 0;trainingExampleI < this.trainingInputs.size();++trainingExampleI){
				
				//perform feed forward
				ArrayList<Double> computedOutput = this.feedForward.executeAndReturnResult(trainingInputs.get(trainingExampleI));
				
				
				//compute errors on output layer
				ArrayList<Double> deltaWO = new ArrayList<Double>();
				for(int outputIndex = 0 ; outputIndex < computedOutput.size(); ++outputIndex){
					Double delta = trainingOutputs.get(trainingExampleI).get(outputIndex)-computedOutput.get(outputIndex);
					Double sigDerivative = computedOutput.get(outputIndex)*(1-computedOutput.get(outputIndex));
					deltaWO.add(delta*sigDerivative);
					totalMSE += (delta*delta);
					errorPerIteration += (delta*delta);
				}
				
				//compute errors on hidden layer
				ArrayList<Double> deltaWH = new ArrayList<Double>();
				for(int hiddenIndex = 0;hiddenIndex < neuralNetwork.getNumberOfHiddenNodes();++hiddenIndex){
					Double delta = 0.0;
					Double sigDerivative = feedForward.getNetHidden(hiddenIndex)*(1-feedForward.getNetHidden(hiddenIndex));
					for(int outputIndex = 0 ; outputIndex < neuralNetwork.getNumberOfOutputNodes(); ++outputIndex){
						delta += deltaWO.get(outputIndex)*neuralNetwork.getHiddenOutputWeight(hiddenIndex , outputIndex);
					}
					deltaWH.add(delta*sigDerivative);
				}
				
				//update hidden-output layer weights
				for(int hiddenIndex = 0;hiddenIndex < neuralNetwork.getNumberOfHiddenNodes();++hiddenIndex){
					for(int outputIndex = 0 ; outputIndex < neuralNetwork.getNumberOfOutputNodes();++outputIndex){
						Double oldW = neuralNetwork.getHiddenOutputWeight(hiddenIndex, outputIndex) ;
						Double newW = oldW + learningRate*deltaWO.get(outputIndex)*feedForward.getNetHidden(hiddenIndex);
						neuralNetwork.updateHiddenOutputWeight(hiddenIndex, outputIndex, newW);
					}
				}
				//update output bias
				for(int outputIndex = 0 ; outputIndex < neuralNetwork.getNumberOfOutputNodes();++outputIndex){
					Double oldBias = neuralNetwork.getOutputBias(outputIndex);
					Double newBias = oldBias + learningRate*deltaWO.get(outputIndex);
					neuralNetwork.updateOutputBias(outputIndex, newBias);
				}
				

				//update input-hidden layer weights
				for(int inputIndex=0;inputIndex < neuralNetwork.getNumberOfInputNodes();++inputIndex){
					for(int hiddenIndex = 0;hiddenIndex < neuralNetwork.getNumberOfHiddenNodes();++hiddenIndex){
						Double oldW = neuralNetwork.getInputHiddenWeight(inputIndex, hiddenIndex);
						Double delta = deltaWH.get(hiddenIndex);
						Double xi = trainingInputs.get(trainingExampleI).get(inputIndex);
						Double newW = oldW + learningRate*delta*xi;
						neuralNetwork.updateInputHiddenWeight(inputIndex, hiddenIndex, newW);
					}
				}
				//update hidden bias
				for(int hiddenIndex = 0;hiddenIndex < neuralNetwork.getNumberOfHiddenNodes();++hiddenIndex){
					Double oldBias = neuralNetwork.getHiddenBias(hiddenIndex);
					Double newBias = oldBias + learningRate*deltaWH.get(hiddenIndex);
					neuralNetwork.updateHiddenBias(hiddenIndex, newBias);
				}
				
			}
			if(errorPerIteration < mnError){
				bestYet = neuralNetwork;
				mnError = errorPerIteration;
			}
		}
		neuralNetwork = bestYet;
		this.totalMSE /= 2.0;
	}

}