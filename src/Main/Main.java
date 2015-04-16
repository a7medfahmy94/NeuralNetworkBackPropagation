package Main;

import java.util.ArrayList;
import java.util.Collections;

import ActivationFunctions.Sigmoid;
import IO.TrainingIO;
import NeuralNet.NeuralNet;
import PropagationAlgorithms.BackPropagation;
import PropagationAlgorithms.FeedForward;

public class Main {
	private static Integer numberOfTrainingIterations = 500;

	private static Integer numberOfHiddenNodes = 25;
	private static Double learningRate = 0.8;
	private static Double gamma = 0.3;
	
	private static TrainingIO io = new TrainingIO("train.txt");
	private static NeuralNet neuralNetwork = new NeuralNet(io.getNumberOfInputs(),numberOfHiddenNodes, io.getNumberOfOutputs());

	public static void main(String[] args) {
		ArrayList<ArrayList<Double> > testInput , testOutput;
		ArrayList<ArrayList<Double> > trainInput , trainOutput ;
		
		int beg = 0 , end = 36 , step = 36;
		double validationError = 0.0 , trainError = 0.0;
		for(int sh = 0 ; sh < 5 ; ++sh){
			testInput = new ArrayList<ArrayList<Double>>();
			testOutput = new ArrayList<ArrayList<Double>>();
			trainInput = new ArrayList<ArrayList<Double>>();
			trainOutput = new ArrayList<ArrayList<Double>>();

			for(int i = beg ; i < end; ++i){
				testInput.add(io.getTrainingInputs().get(i));
				testOutput.add(io.getTrainingOutputs().get(i));
			}
			for(int i = 0 ; i < io.getNumberOfTrainingExamples() ; ++i){
				if(i >= beg && i < end)continue;
				trainInput.add(io.getTrainingInputs().get(i));
				trainOutput.add(io.getTrainingOutputs().get(i));
			}
			trainError += run(trainInput,trainOutput);
			validationError += testFeedForward(testInput , testOutput);
			beg += step;
			end += step;
		}
		trainError /= 5.0;
		validationError /= 5.0;
		System.out.printf("train error = %.2f\n" , trainError);
		System.out.printf("validation error = %.2f" , validationError);
	}
	
	public static double testFeedForward(ArrayList<ArrayList<Double> > testInput,ArrayList<ArrayList<Double> > testOutput){
		FeedForward ff = new FeedForward(neuralNetwork, new Sigmoid(gamma));
		Double e = 0.0;
		for(int i = 0 ; i < testInput.size(); ++i){
			ArrayList<Double> out = ff.executeAndReturnResult(testInput.get(i));
			for(int j = 0 ; j < out.size(); ++j){
				e += (out.get(j) - testOutput.get(i).get(j))*(out.get(j) - testOutput.get(i).get(j));
			}
		}
		return e/2.0;
	}
	public static double run(ArrayList<ArrayList<Double> > trainInput,ArrayList<ArrayList<Double> > trainOutput){
		
				
		BackPropagation backPropagation = new BackPropagation(neuralNetwork, numberOfTrainingIterations,gamma);
		backPropagation.setLearningRate(learningRate);
		backPropagation.executeBackPropagation(trainInput,trainOutput);
		
		return backPropagation.getTotalMSE();
	}

}