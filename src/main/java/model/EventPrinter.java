package model;

import org.apache.log4j.Logger;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.events.LearningEventType;
import org.neuroph.core.learning.SupervisedLearning;

/**
 * Created by DiKey on 14.08.2015.
 */
public class EventPrinter implements LearningEventListener {

    public static Logger logger = Logger.getLogger(EventPrinter.class);

    public void handleLearningEvent(LearningEvent event) {

        if (!(event.getSource() instanceof SupervisedLearning)) {
            logger.info(event.toString());
            return;
        }

        SupervisedLearning learningRule = (SupervisedLearning) event.getSource();
        int i = learningRule.getCurrentIteration();

        logger.info("iteration: " + i + "; totalNetworkError: " + learningRule.getTotalNetworkError());

        if (i % 10 == 0) {
            synchronized (learningRule.getNeuralNetwork()) {
                calculateAndWrite(learningRule.getNeuralNetwork(), "network.data_" + i);

            }
        }

        if (LearningEventType.LEARNING_STOPPED.equals(event.getEventType())) {
            synchronized (learningRule.getNeuralNetwork()) {

                calculateAndWrite(learningRule.getNeuralNetwork(), "network.data");
                learningRule.getNeuralNetwork().notify();
            }
        }
    }

    private void calculateAndWrite(NeuralNetwork net, String fileName) {
        net.calculate();

        logger.info("Writing net to file " + fileName);
        net.save(Runner.dataPath + fileName);
        logger.info("Writing finished");
    }
}
