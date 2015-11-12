package methods;

import model.Log;
import networks.NetworkBuilder;
import org.encog.ml.BasicML;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.BasicTraining;

/**
 * Created by ledenev.p on 18.08.2015.
 */
public abstract class TrainingBuilder {

    private BasicTraining training;

    private NetworkBuilder networkBuilder;

    public TrainingBuilder(NetworkBuilder networkBuilder) {
        this.networkBuilder = networkBuilder;
    }

    public BasicTraining build(MLDataSet trainingSet) {

        training = createTraining(networkBuilder.build(trainingSet), trainingSet);

        Log.info("Training with " + training.getClass().getSimpleName());

        return training;
    }

    public void saveNetwork() {
        networkBuilder.saveNetwork();
    }

    public void setTrainingDataTitle(String trainingSetName) {
        networkBuilder.setTrainingSetName(trainingSetName);
    }

    protected abstract BasicTraining createTraining(BasicML network, MLDataSet trainingSet);

    public BasicML getNetwork() {
        return networkBuilder.getNetwork();
    }
}
