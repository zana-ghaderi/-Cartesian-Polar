/*
 * This source file contains the Verilog description of an example
 * HW/SW interface for the Xilinx XUP board.  
 * 
 * Redistributions of any form whatsoever must retain and/or include the
 * following acknowledgment, notices and disclaimer:
 *
 * This product includes software developed by Carnegie Mellon University.
 *
 * Copyright (c) 2006 by J. C. Hoe, Carnegie Mellon University
 *
 * You may not use the name "Carnegie Mellon University" or derivations
 * thereof to endorse or promote products derived from this software.
 *
 * If you modify the software you must place a notice on or within any
 * modified version provided or made available to any third party stating
 * that you have modified the software.  The notice shall include at least
 * your name, address, phone number, email address and the date and purpose
 * of the modification.
 *
 * THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER
 * EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO ANY WARRANTY
 * THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS OR BE ERROR-FREE AND ANY
 * IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
 * TITLE, OR NON-INFRINGEMENT.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
 * BE LIABLE FOR ANY DAMAGES, INCLUDING BUT NOT LIMITED TO DIRECT, INDIRECT,
 * SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN
 * ANY WAY CONNECTED WITH THIS SOFTWARE (WHETHER OR NOT BASED UPON WARRANTY,
 * CONTRACT, TORT OR OTHERWISE).
 */

module refhw
  (
    Clk, // assume synchronous with BRAM_Clk_A
    Rst, // system reset

    // BRAM-like interface for PPC405's OCM controller
    BRAM_Rst_A,
    BRAM_Clk_A,
    BRAM_EN_A,
    BRAM_WEN_A,
    BRAM_Addr_A,
    BRAM_Din_A,
    BRAM_Dout_A
  );

  input Clk;
  input Rst;

  input BRAM_Rst_A;
  input BRAM_Clk_A;
  input BRAM_EN_A;
  input [3:0] BRAM_WEN_A;
  input [31:0] BRAM_Addr_A;
  output reg [31:0] BRAM_Din_A;
  input [31:0] BRAM_Dout_A;

  ////////////////////////////////////////////////////////////
  // four 4KByte memory-mapped regions
  // 00: input circular buffer
  // 01: output circular buffer
  // 10: control registers
  // 11: OCM memory (optional)
  ////////////////////////////////////////////////////////////
  wire [31:0] readOcmIn;
  wire [31:0] readOcmOut;
  reg [31:0] readReg;
  wire [31:0] readOcmMem;

  reg [31:0] BRAM_Addr_ALast;
  always@( posedge BRAM_Clk_A ) begin
    BRAM_Addr_ALast<=BRAM_Addr_A;
  end

  always@ * begin
        case (BRAM_Addr_ALast[13:12])
          2'b00: BRAM_Din_A = readOcmIn;
          2'b01: BRAM_Din_A = readOcmOut;
          2'b10: BRAM_Din_A = readReg;
          2'b11: BRAM_Din_A = readOcmMem;
        endcase
  end

  ////////////////////////////////////////////////////////////
  // cntrl/status registers memory-mapped to the 3rd 4KByte region
  ////////////////////////////////////////////////////////////

  // example cntrl/status registers
  reg [9:0] ibufHead;
  reg [9:0] ibufTail;
  reg [9:0] obufHead;
  reg [9:0] obufTail;
  reg sftRst;
 
  wire [9:0] obufHeadNext;
  assign obufHeadNext=obufHead+1;
  
  always@(posedge BRAM_Clk_A) begin
    // control register write
    // affects ibufHead, obufTail, sftRst

    if (BRAM_EN_A&&BRAM_WEN_A[2]) begin
      if (BRAM_Addr_A[13:12]==2'b10) begin
        case (BRAM_Addr_A[7:5])
          3'b000: ibufHead<=BRAM_Dout_A;
          3'b011: obufTail<=BRAM_Dout_A;
          3'b111: sftRst<=BRAM_Dout_A;
        endcase
      end
    end
  end
 
  always @(posedge BRAM_Clk_A) begin
    // control register read
    case (BRAM_Addr_A[7:5]) 
        3'b000: readReg <= ibufHead;
        3'b001: readReg <= ibufTail;
        3'b010: readReg <= obufHead;
        3'b011: readReg <= obufTail;
        3'b111: readReg <= {31'h00000000,sftRst};
        default: readReg <= 32'hxxxxxxxx;
    endcase
  end

  ////////////////////////////////////////////////////////////
  // example scratch-pad memory (based on BRAM) memory-mapped to the 4th 4KByte region
  ////////////////////////////////////////////////////////////

  DPSRAM4KBY4 ocmscratch (
    .BRAM_Rst_A(BRAM_Rst_A),
    .BRAM_Clk_A(BRAM_Clk_A),
    .BRAM_EN_A(BRAM_EN_A),
    .BRAM_WEN_A(BRAM_WEN_A[3] && (BRAM_Addr_A[13:12]==2'b11)),
    .BRAM_Addr_A(BRAM_Addr_A[11:2]),
    .BRAM_Din_A(BRAM_Dout_A),
    .BRAM_Dout_A(readOcmMem),

    // The 2nd port is not used
    .BRAM_Rst_B(1'b0),
    .BRAM_Clk_B(1'b0),
    .BRAM_EN_B(1'b0),
    .BRAM_WEN_B(1'b0),
    .BRAM_Addr_B(10'b0),
    .BRAM_Din_B(32'b0),
    .BRAM_Dout_B()
  );

  ////////////////////////////////////////////////////////////
  // example IN and OUT circular memory buffer in the 1st and 2nd 4KByte region
  // 
  // CPU writes into ibuf at ibufhead and advances ibufhead (all by mmaped writes)
  // When ibufhead!=ibuftail the entries between them in ibuf are new data.
  // When the HW module consumes from ibuftail, it advances ibuftail (read by the CPU) 
  // to free the entries.  Similarly, the HW module writes into obuf 
  // at obufhead in a mirrored protocol to make data available to the CPU.
  //
  // In this example, the HW module is just a loop-back FIFO that consumes
  // from ibuf whenever something new is available in ibuf (ibufhead!=ibuftail)
  // and the FIFO is not full.  The FIFO writes back out to obuf (in FIFO order) 
  // whenever the FIFO is not empty and the obuf is not full (obufhead+1!=obuftail).
  ////////////////////////////////////////////////////////////

  wire [31:0] ibufData;

  // signals to the loopback fifo for demo'ing the circular buffer interface
  wire [31:0] eFifoDout;
  wire eFifoFull;
  wire eFifoEmpty;
  wire eFifoEnq;
  wire eFifoDeq;

  assign eFifoEnq=(!eFifoFull)&&(ibufHead!=ibufTail);

  always@(posedge Clk) begin
    if (sftRst) begin
      ibufTail<=0;
    end else begin
      if (eFifoEnq) begin
        ibufTail<=ibufTail+1;
      end
    end
  end

  // incoming circular buffer
  DPSRAM4KBY4 ibuf (
    // memory-mapped on OCM for CPU write
    .BRAM_Rst_A(BRAM_Rst_A),
    .BRAM_Clk_A(BRAM_Clk_A),
    .BRAM_EN_A(BRAM_EN_A),
    .BRAM_WEN_A(BRAM_WEN_A[0] && (BRAM_Addr_A[13:12]==2'b00)),
    .BRAM_Addr_A(BRAM_Addr_A[11:2]),
    .BRAM_Din_A(BRAM_Dout_A),
    .BRAM_Dout_A(readOcmIn),
    
    // read by the HW module
    .BRAM_Rst_B(sftRst),
    .BRAM_Clk_B(Clk),
    .BRAM_EN_B(1'b1),
    .BRAM_WEN_B(1'b0),
    .BRAM_Addr_B(ibufTail),
    .BRAM_Din_B(32'b0),
    .BRAM_Dout_B(ibufData)
  );

  assign eFifoDeq=(!eFifoEmpty)&&(obufHeadNext!=obufTail);
  always@(posedge Clk) begin
    if (sftRst) begin
      obufHead<=0;
    end else begin
      if (eFifoDeq) begin
        obufHead<=obufHead+1;
      end
    end
  end

  // outgoing circular buffer
  DPSRAM4KBY4 obuff (
    // memory-mapped on OCM for CPU read
    .BRAM_Rst_A(BRAM_Rst_A),
    .BRAM_Clk_A(BRAM_Clk_A),
    .BRAM_EN_A(BRAM_EN_A),
    .BRAM_WEN_A(BRAM_WEN_A[1] && (BRAM_Addr_A[13:12]==2'b01)),
    .BRAM_Addr_A(BRAM_Addr_A[11:2]),
    .BRAM_Din_A(BRAM_Dout_A),
    .BRAM_Dout_A(readOcmOut),

    // written by the HW module
    .BRAM_Rst_B(sftRst),
    .BRAM_Clk_B(Clk),
    .BRAM_EN_B(1'b1),
    .BRAM_WEN_B(eFifoDeq),
    .BRAM_Addr_B(obufHead),
    .BRAM_Din_B(eFifoDout),
    .BRAM_Dout_B()
  );

  // signals delayed to match synchronous SRAM timing of ibuf
  reg eFifoEnqLast;
  always@(posedge Clk) begin
    eFifoEnqLast<=eFifoEnq;  
  end

  // loop-back fifo defined below.
  FIFO E (
   .Dout(eFifoDout), 
   .Deq(eFifoDeq), 
   .Din(ibufData), 
   .Enq(eFifoEnqLast), 
   .AlmostFull(eFifoFull), 
   .Empty(eFifoEmpty), 
   .Rst(sftRst),
   .Clk(Clk)
  );

endmodule

////////////////////////////////////////////////////////////////////
// Dual-ported synchronous memory module inferred as BRAM
////////////////////////////////////////////////////////////////////
module DPSRAM4KBY4 (
    input BRAM_Rst_A, // not used
    input BRAM_Clk_A,
    input BRAM_EN_A,
    input BRAM_WEN_A,
    input [9:0] BRAM_Addr_A,
    input [31:0] BRAM_Din_A, // I/O relavetive to BRAM 
    output reg [31:0] BRAM_Dout_A, 

    input BRAM_Rst_B, // not used
    input BRAM_Clk_B,
    input BRAM_EN_B,
    input BRAM_WEN_B,
    input [9:0] BRAM_Addr_B,
    input [31:0] BRAM_Din_B, // I/O relavetive to BRAM 
    output reg [31:0] BRAM_Dout_B 
  );

  reg [31:0] mem[0:1023];

  always @(posedge BRAM_Clk_A) begin
    if (BRAM_EN_A && BRAM_WEN_A) begin
        mem[BRAM_Addr_A] <= BRAM_Din_A;
    end
    BRAM_Dout_A <= mem[BRAM_Addr_A];
  end

  always @(posedge BRAM_Clk_B) begin
    if (BRAM_EN_B && BRAM_WEN_B) begin
        mem[BRAM_Addr_B] <= BRAM_Din_B;
    end
    BRAM_Dout_B <= mem[BRAM_Addr_B];
  end

endmodule


////////////////////////////////////////////////////////////////////
// This FIFO is used for loop-back in this demonstration.
// To keep it simple, you should try to use the same simple FIFO interface 
// semantics for your HW accelerator.  If you are adventurous, you can build
// any memory-mapped interface to connnect directly to the OCM interface.
// (See the reading and writing of control registers for example.)
////////////////////////////////////////////////////////////////////

module FIFO(
   output [31:0] Dout, 
   input Deq, 
   input [31:0] Din, 
   input Enq, 
   output AlmostFull, 
   output Empty, 
   input Rst,
   input Clk);

   reg [31:0] mem [0:63];
   
   reg [5:0] head, tail;
   reg [5:0] headNext, headNextNext;

   // Normally, "full=(headNext==tail)"
   // The strangeness below is to deal with the synchronous read 
   // of ibuf. The copying from ibuf to fifo checks fullness
   // one cycle before the actual push.

   assign AlmostFull=((headNext==tail)||headNextNext==tail);  

   assign Empty=(head==tail);
   assign Dout=mem[tail];

   always@(posedge Clk) begin
     if (Rst) begin
       head<=0;
       headNext<=1;
       headNextNext<=2;
       tail<=0;
     end else begin
       if (Enq) begin
         head<=head+1;
         headNext<=headNext+1;
         headNextNext<=headNextNext+1;
       end
       if (Deq) begin
         tail<=tail+1;
       end
     end
   end

   always@(posedge Clk) begin
     if (Enq) begin
        mem[head]<=Din;
     end
   end

endmodule
