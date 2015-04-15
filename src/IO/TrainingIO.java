package IO;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class TrainingIO {
	private Scanner fileInput;
	private File trainingFile;
	private ArrayList<ArrayList<Double> > trainingInputs;
	private ArrayList<ArrayList<Double> > trainingOutputs;
	private Integer numberOfInputs;
	private Integer numberOfOutputs;
	private Integer numberOfTrainingExamples;
	
	public TrainingIO(String fileName){
		trainingFile = new File(fileName);
		trainingInputs = new ArrayList<ArrayList<Double>>();
		trainingOutputs = new ArrayList<ArrayList<Double>>();
		try{
			read();
		}catch(IOException e){
			e.printStackTrace();
		}
	}

	public ArrayList<ArrayList<Double> > getTrainingInputs(){
		
		return trainingInputs;
	}
	
	public ArrayList<ArrayList<Double> > getTrainingOutputs(){
		
		return trainingOutputs;
	}

	public Integer getNumberOfInputs(){
		return numberOfInputs;
	}

	public Integer getNumberOfOutputs(){
		return numberOfOutputs;
	}

	public Integer getNumberOfTrainingExamples(){
		return numberOfTrainingExamples;
	}
	
	public void read() throws IOException{
		fileInput = new Scanner(trainingFile);

		numberOfInputs = fileInput.nextInt();
		numberOfOutputs = fileInput.nextInt();
		numberOfTrainingExamples = fileInput.nextInt();

		for(int T = 0 ; T < numberOfTrainingExamples; ++T){
			trainingInputs.add(new ArrayList<Double>());
			trainingOutputs.add(new ArrayList<Double>());
			for(int I = 0 ; I < numberOfInputs; ++I){
				trainingInputs.get(T).add(fileInput.nextDouble());
			}
			for(int O = 0 ; O < numberOfOutputs; ++O){
				trainingOutputs.get(T).add(fileInput.nextDouble());
			}
		}
		
		fileInput.close();
	}
}
