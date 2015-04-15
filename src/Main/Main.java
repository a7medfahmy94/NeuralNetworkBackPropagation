package Main;

import java.util.ArrayList;

import ActivationFunctions.Sigmoid;
import IO.TrainingIO;
import NeuralNet.NeuralNet;
import PropagationAlgorithms.BackPropagation;
import PropagationAlgorithms.FeedForward;

public class Main {
	private static Integer numberOfHiddenNodes = 20;
	private static Integer numberOfTrainingIterations = 10000;
	private static Double learningRate = 0.9;

	
	public static void main(String[] args) {
		run();
	}
	
	public static void run(){
		TrainingIO io = new TrainingIO("train.txt");
		
		ArrayList<ArrayList<Double> > trainInput , trainOutput , testInput , testOutput;
		trainInput = new ArrayList<ArrayList<Double>>();
		trainOutput = new ArrayList<ArrayList<Double>>();
		testInput = new ArrayList<ArrayList<Double>>();
		testOutput = new ArrayList<ArrayList<Double>>();
		for(int i = 0 ; i < 100 ; ++i){
			trainInput.add(io.getTrainingInputs().get(i));
			trainOutput.add(io.getTrainingOutputs().get(i));
		}
		for(int i = 100 ; i < io.getNumberOfTrainingExamples(); ++i){
			testInput.add(io.getTrainingInputs().get(i));
			testOutput.add(io.getTrainingOutputs().get(i));
		}
		
		
		NeuralNet neuralNetwork = new NeuralNet(io.getNumberOfInputs(),numberOfHiddenNodes, io.getNumberOfOutputs());

		FeedForward ff = new FeedForward(neuralNetwork, new Sigmoid(1.0));
		Double e = 0.0;
		for(int i = 0 ; i < testInput.size(); ++i){
			ArrayList<Double> out = ff.executeAndReturnResult(testInput.get(i));
			for(int j = 0 ; j < out.size(); ++j){
				e += Math.abs(out.get(j) - testOutput.get(i).get(j));
			}
		}
		System.out.println(e);
		e = 0.0;

		
		
		BackPropagation backPropagation = new BackPropagation(neuralNetwork, numberOfTrainingIterations);
		backPropagation.setLearningRate(learningRate);
		backPropagation.executeBackPropagation(testInput,testOutput);

		
		for(int i = 0 ; i < testInput.size(); ++i){
			ArrayList<Double> out = ff.executeAndReturnResult(testInput.get(i));
			for(int j = 0 ; j < out.size(); ++j){
				e += Math.abs(out.get(j) - testOutput.get(i).get(j));
			}
		}
		System.out.println(e);
		
	}
	public void xor(){
		ArrayList<ArrayList<Double> > in = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double> > out = new ArrayList<ArrayList<Double>>();
		in.add(new ArrayList<Double>());
		in.add(new ArrayList<Double>());
		in.add(new ArrayList<Double>());
		in.add(new ArrayList<Double>());
		out.add(new ArrayList<Double>());
		out.add(new ArrayList<Double>());
		out.add(new ArrayList<Double>());
		out.add(new ArrayList<Double>());
		
		in.get(0).add(0.0);
		in.get(0).add(0.0);
		out.get(0).add(0.0);

		in.get(1).add(0.0);
		in.get(1).add(1.0);
		out.get(1).add(1.0);

		in.get(2).add(1.0);
		in.get(2).add(0.0);
		out.get(2).add(1.0);
		
		in.get(3).add(1.0);
		in.get(3).add(1.0);
		out.get(3).add(0.0);
		
		NeuralNet nn = new NeuralNet(2, 2, 1);
		BackPropagation bp = new BackPropagation(nn, numberOfTrainingIterations);
		bp.setLearningRate(learningRate);
		
		nn.printNeuralNetwork();
		bp.executeBackPropagation(in, out);
		nn.printNeuralNetwork();
		
		FeedForward ff = new FeedForward(nn, new Sigmoid(1.0));
		ArrayList<Double> test = new ArrayList<Double>();
		test.add(1.0);
		test.add(1.0);
		System.out.println(ff.executeAndReturnResult(test).get(0));

	}

}