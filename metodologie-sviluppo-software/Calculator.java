public class Calculator {
    public static int Add(String numbersList){

        if(numbersList.equals(""))
            return 0;

        String regex = "[\n]";
        String[] numbers = numbersList.split(regex);

        System.out.println(numbers[0]);

        String delimiter = "";
        if(numbers[0].startsWith("//")){
            delimiter = numbers[0].substring(2);
            numbersList = numbersList.replace(numbers[0], "");
        }
        System.out.println(delimiter);
        System.out.println(numbersList);

        regex = String.format("[,%s\n]", delimiter);
        numbers = numbersList.split(regex);
        System.out.println(numbers.length);
        int sum = 0;

        for(String number : numbers){

            sum = Integer.parseInt(number) + sum;
        }

        return sum;
    }
}
