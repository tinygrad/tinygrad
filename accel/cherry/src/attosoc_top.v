module top (
  input clk_i,
  input [3:0] sw,
  output [11:0] led,
  output ser_tx,
  input  ser_rx,
);

    //assign led = {&sw, |sw, ^sw, ~^sw};

    reg clk50 = 1'b0;
    always @(posedge clk_i)
        clk50 <= ~clk50;

    wire clk;
    BUFGCTRL bufg_i (
        .I0(clk50),
        .CE0(1'b1),
        .S0(1'b1),
        .O(clk)
    );


  //  wire clk = clk_i;

    //reg clkdiv;
    //reg [22:0] ctr;
    //always @(posedge clk) {clkdiv, ctr} <= ctr + 1'b1;

    wire [7:0] soc_led;
    attosoc soc_i(
        .clk(clk),
        .reset(sw[0]),
        .led(soc_led),
        .ser_tx(ser_tx),
        .ser_rx(ser_rx),
    );

    // this maps 2 bits to each LED
    generate
        genvar i;
        for (i = 0; i < 4; i++) begin
            assign led[0 + i] = soc_led[2 * i]; // R
            assign led[4 + i] = soc_led[(2 * i) + 1]; // G
            assign led[8 + i] = &soc_led[2 * i +: 2]; // B
        end
    endgenerate
endmodule
