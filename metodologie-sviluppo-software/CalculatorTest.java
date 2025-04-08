import static org.junit.Assert.*;
import org.junit.*;
import java.util.ArrayList;

public class CalculatorTest{
    
    @Test
    public void noNumber(){
        assertEquals(Calculator.Add(""),0);
    }

    @Test
    public void oneNumber(){
        assertEquals(Calculator.Add("5"),5);
    }

    @Test
    public void twoNumber(){
        assertEquals(Calculator.Add("5,7"),12);
    }

    @Test
    public void nNumber(){
        int randomSize = (int)(Math.random() * 21);
        
        int sum = 0;
        StringBuilder sb = new StringBuilder();
        
        for(int i = 0; i < randomSize; i++){

            int randomNum = (int)(Math.random() * 101);
            sum = sum + randomNum;

            sb.append(randomNum);
            if (i < randomSize - 1) {
                sb.append(",");
            }
        }

        assertEquals(Calculator.Add(sb.toString()), sum);
    }

    @Test
    public void newLineAndComma(){
        assertEquals(Calculator.Add("5\n7,9"),21);
    }

    @Test
    public void arbitraryDelimiter(){
        String input = "//p\n5p9p3,6";
        assertEquals(Calculator.Add(input), 23);
    }
}